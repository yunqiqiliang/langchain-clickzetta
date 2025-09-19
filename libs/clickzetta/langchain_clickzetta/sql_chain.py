"""ClickZetta SQL Chain for LangChain integration."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from langchain_clickzetta.engine import ClickZettaEngine

logger = logging.getLogger(__name__)

# Default prompt template for SQL generation
DEFAULT_SQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"],
    template="""Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for the relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

Question: {input}""",
)

# Default prompt for query result formatting
DEFAULT_RESULT_PROMPT = PromptTemplate(
    input_variables=["input", "sql_query", "sql_result"],
    template="""Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {input}
SQL Query: {sql_query}
SQL Result: {sql_result}
Answer: """,
)


class ClickZettaSQLChain(BaseModel, Runnable):
    """Chain for querying ClickZetta database with natural language.

    This chain takes a natural language question, converts it to SQL,
    executes it against ClickZetta, and returns a natural language answer.
    """

    name: str = Field(default="ClickZettaSQLChain")
    engine: ClickZettaEngine = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)
    sql_prompt: BasePromptTemplate = Field(default=DEFAULT_SQL_PROMPT)
    result_prompt: BasePromptTemplate = Field(default=DEFAULT_RESULT_PROMPT)
    table_names: list[str] | None = None
    top_k: int = 10
    return_sql: bool = False
    return_intermediate_steps: bool = False
    use_query_checker: bool = True
    query_checker_prompt: BasePromptTemplate | None = None
    verbose: bool = False

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def validate_inputs(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate that required inputs are provided."""
        if "engine" not in values:
            raise ValueError("ClickZettaEngine must be provided")
        if "llm" not in values:
            raise ValueError("Language model must be provided")
        return values

    @property
    def input_keys(self) -> list[str]:
        """Return the singular input key."""
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return the output keys."""
        _output_keys = [self.output_key]
        if self.return_sql:
            _output_keys.append("sql_query")
        if self.return_intermediate_steps:
            _output_keys.append("intermediate_steps")
        return _output_keys

    @property
    def input_key(self) -> str:
        """Return the singular input key."""
        return "query"

    @property
    def output_key(self) -> str:
        """Return the singular output key."""
        return "result"

    def _get_table_info(self) -> str:
        """Get information about the database tables."""
        return self.engine.get_table_info(self.table_names)

    def _generate_sql_query(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> str:
        """Generate SQL query from natural language question."""
        table_info = self._get_table_info()

        prompt_inputs = {
            "input": inputs[self.input_key],
            "table_info": table_info,
            "dialect": "ClickZetta",
            "top_k": self.top_k,
        }

        sql_prompt_value = self.sql_prompt.format_prompt(**prompt_inputs)

        if run_manager:
            run_manager.on_text("SQL Prompt:", end="\n", verbose=self.verbose)
            run_manager.on_text(
                sql_prompt_value.to_string(), end="\n", verbose=self.verbose
            )

        # Generate SQL using LLM
        try:
            # Try using invoke first (for modern LLMs)
            sql_query = self.llm.invoke(sql_prompt_value.to_string()).strip()
        except Exception as invoke_error:
            try:
                # Fallback to generate with string input
                sql_response = self.llm.generate([sql_prompt_value.to_string()])
                sql_query = sql_response.generations[0][0].text.strip()
            except Exception as generate_error:
                logger.error(f"LLM invoke failed: {invoke_error}")
                logger.error(f"LLM generate failed: {generate_error}")
                raise generate_error

        # Extract SQL query from response (in case LLM includes extra text)
        if "SQLQuery:" in sql_query:
            sql_query = sql_query.split("SQLQuery:")[1].split("SQLResult:")[0].strip()

        # Remove surrounding quotes and backticks if present
        sql_query = sql_query.strip('"').strip("'").strip("`").strip()

        return sql_query

    def _execute_sql_query(
        self,
        sql_query: str,
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> str:
        """Execute SQL query against ClickZetta."""
        if run_manager:
            run_manager.on_text("SQL Query:", end="\n", verbose=self.verbose)
            run_manager.on_text(sql_query, end="\n", verbose=self.verbose)

        try:
            results, columns = self.engine.execute_query(sql_query)

            # Format results as string
            if not results:
                return "No results found."

            # Limit results to top_k
            limited_results = results[: self.top_k]

            # Format as table-like string
            result_str = []
            if columns:
                result_str.append(" | ".join(columns))
                result_str.append("-" * len(" | ".join(columns)))

            for row in limited_results:
                if isinstance(row, dict):
                    values = [str(row.get(col, "")) for col in columns]
                else:
                    values = [str(val) for val in row]
                result_str.append(" | ".join(values))

            formatted_result = "\n".join(result_str)

            if run_manager:
                run_manager.on_text("SQL Result:", end="\n", verbose=self.verbose)
                run_manager.on_text(formatted_result, end="\n", verbose=self.verbose)

            return formatted_result

        except Exception as e:
            error_msg = f"Error executing SQL query: {str(e)}"
            logger.error(error_msg)
            if run_manager:
                run_manager.on_text(error_msg, end="\n", verbose=self.verbose)
            return error_msg

    def _format_result(
        self,
        inputs: dict[str, Any],
        sql_query: str,
        sql_result: str,
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> str:
        """Format the final result using LLM."""
        prompt_inputs = {
            "input": inputs[self.input_key],
            "sql_query": sql_query,
            "sql_result": sql_result,
        }

        result_prompt_value = self.result_prompt.format_prompt(**prompt_inputs)

        if run_manager:
            run_manager.on_text("Result Prompt:", end="\n", verbose=self.verbose)
            run_manager.on_text(
                result_prompt_value.to_string(), end="\n", verbose=self.verbose
            )

        # Generate final answer using LLM
        try:
            # Try using invoke first (for modern LLMs)
            final_result = self.llm.invoke(result_prompt_value.to_string()).strip()
        except Exception as invoke_error:
            try:
                # Fallback to generate with string input
                result_response = self.llm.generate([result_prompt_value.to_string()])
                final_result = result_response.generations[0][0].text.strip()
            except Exception as generate_error:
                logger.error(f"LLM invoke failed: {invoke_error}")
                logger.error(f"LLM generate failed: {generate_error}")
                raise generate_error

        return final_result

    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Execute the SQL chain."""
        return self._call(input)

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Execute the SQL chain."""
        intermediate_steps = []

        # Step 1: Generate SQL query
        try:
            sql_query = self._generate_sql_query(inputs, run_manager)
            intermediate_steps.append({"step": "sql_generation", "query": sql_query})
        except Exception as e:
            error_msg = f"Error generating SQL query: {str(e)}"
            logger.error(error_msg)
            return {self.output_key: error_msg}

        # Step 2: Execute SQL query
        try:
            sql_result = self._execute_sql_query(sql_query, run_manager)
            intermediate_steps.append({"step": "sql_execution", "result": sql_result})
        except Exception as e:
            error_msg = f"Error executing SQL query: {str(e)}"
            logger.error(error_msg)
            return {self.output_key: error_msg}

        # Step 3: Format final result
        try:
            final_result = self._format_result(
                inputs, sql_query, sql_result, run_manager
            )
            intermediate_steps.append(
                {"step": "result_formatting", "result": final_result}
            )
        except Exception as e:
            logger.error(f"Error formatting result: {str(e)}")
            # Fallback to raw SQL result
            final_result = sql_result

        # Prepare output
        output = {self.output_key: final_result}

        if self.return_sql:
            output["sql_query"] = sql_query

        if self.return_intermediate_steps:
            output["intermediate_steps"] = intermediate_steps

        return output

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "clickzetta_sql_chain"

    @classmethod
    def from_engine(
        cls,
        engine: ClickZettaEngine,
        llm: BaseLanguageModel,
        **kwargs: Any,
    ) -> ClickZettaSQLChain:
        """Create ClickZettaSQLChain from ClickZettaEngine.

        Args:
            engine: ClickZetta database engine
            llm: Language model for SQL generation and result formatting
            **kwargs: Additional arguments to pass to the chain

        Returns:
            ClickZettaSQLChain instance
        """
        return cls(engine=engine, llm=llm, **kwargs)
