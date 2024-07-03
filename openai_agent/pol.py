import json
import os
from pprint import pprint
from dotenv import load_dotenv
from langchain_community.tools.polygon.aggregates import PolygonAggregates
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools.polygon.aggregates import PolygonAggregatesSchema

load_dotenv()

os.environ["POLYGON_API_KEY"] = os.getenv('POLYGON_API_KEY')

api_wrapper = PolygonAPIWrapper()
ticker = "X:BTCUSD"

params = PolygonAggregatesSchema(
    ticker=ticker,
    timespan="day",
    timespan_multiplier=1,
    from_date="2023-01-09",
    to_date="2023-02-10",
)

aggregates_tool = PolygonAggregates(api_wrapper=api_wrapper)
aggregates = aggregates_tool.run(tool_input=params.dict())
aggregates_json = json.loads(aggregates)

print(f"Total aggregates: {len(aggregates_json)}")
pprint(f"Aggregates: {aggregates_json}")