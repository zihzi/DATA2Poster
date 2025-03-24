import ast
import base64
import importlib
import io
import os
import re
import traceback
from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd

from lida.datamodel import ChartExecutorResponse, Summary


def preprocess_code(code: str) -> str:
    """Preprocess code to remove any preamble and explanation text"""

    code = code.replace("<imports>", "")
    code = code.replace("<stub>", "")
    code = code.replace("<transforms>", "")

    # remove all text after chart = plot(data)
    if "chart = plot(data)" in code:
        index = code.find("chart = plot(data)")
        if index != -1:
            code = code[: index + len("chart = plot(data)")]

    if "```" in code:
        pattern = r"```(?:\w+\n)?([\s\S]+?)```"
        matches = re.findall(pattern, code)
        if matches:
            code = matches[0]

    if "import" in code:
        # return only text after the first import statement
        index = code.find("import")
        if index != -1:
            code = code[index:]

    code = code.replace("```", "")
    if "chart = plot(data)" not in code:
        code = code + "\nchart = plot(data)"
    return code


def get_globals_dict(code_string, data):
    # Parse the code string into an AST
    tree = ast.parse(code_string)
    # Extract the names of the imported modules and their aliases
    imported_modules = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = importlib.import_module(alias.name)
                imported_modules.append((alias.name, alias.asname, module))
        elif isinstance(node, ast.ImportFrom):
            module = importlib.import_module(node.module)
            for alias in node.names:
                obj = getattr(module, alias.name)
                imported_modules.append(
                    (f"{node.module}.{alias.name}", alias.asname, obj)
                )

    # Import the required modules into a dictionary
    globals_dict = {}
    for module_name, alias, obj in imported_modules:
        if alias:
            globals_dict[alias] = obj
        else:
            globals_dict[module_name.split(".")[-1]] = obj

    ex_dicts = {"pd": pd, "data": data, "plt": plt}
    globals_dict.update(ex_dicts)
    return globals_dict


class ChartExecutor:
    """Execute code and return chart object"""

    def __init__(self) -> None:
        pass

    def execute(
        self,
        code_specs: List[str],
        data: Any,
        summary: Summary,
        library="altair",
        return_error: bool = False,
    ) -> Any:
        """Validate and convert code"""

        if isinstance(summary, dict):
            summary = Summary(**summary)

        charts = []
        code_spec_copy = code_specs[:]
        code_specs = preprocess_code(code_specs)
        try:
                    ex_locals = get_globals_dict(code_specs, data)
                    exec(code_specs, ex_locals)
                    chart = ex_locals["chart"]
                    vega_spec = chart.to_dict()
                    del vega_spec["data"]
                    if "datasets" in vega_spec:
                        del vega_spec["datasets"]
                    vega_spec["data"] = {"values": data.to_dict(orient="records")}
                    charts.append(
                        ChartExecutorResponse(
                            spec=vega_spec,
                            status=True,
                            raster=None,
                            code=code_specs,
                            library="altair",
                        )
                    )
        except Exception as exception_error:
                    print(code_spec_copy, "\n===========\n")
                    print(exception_error)
                    print(traceback.format_exc())
                    if return_error:
                        charts.append(
                            ChartExecutorResponse(
                                spec=None,
                                status=False,
                                raster=None,
                                code=code_spec_copy,
                                library="altair",
                                error={
                                    "message": str(exception_error),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                        )
        return charts


            