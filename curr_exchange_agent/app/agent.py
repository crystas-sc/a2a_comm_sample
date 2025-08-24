import os

from collections.abc import AsyncIterable
from typing import Any, Literal

import httpx

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


from typing_extensions import TypedDict
from typing import Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage # Or HumanMessage, AIMessage, SystemMessage etc.
from langgraph.graph.message import add_messages


# Define the Graph State
class CurrencyConversionState(TypedDict):
    amount: float
    from_currency: str
    to_currency: str
    exchange_rate: Annotated[float, "Exchange rate between from_currency and to_currency"]
    converted_amount: Annotated[float, "Converted amount in the target currency"]
    error_message: Annotated[str, "Error message if any issue arises"]
    messages: Annotated[List[BaseMessage], add_messages] 
    structured_response: ResponseFormat


# Create the Nodes
def _get_exchange_rate(state: CurrencyConversionState) -> CurrencyConversionState:
    print("---Node: Getting Exchange Rate---")
    print("--------state------")
    print(state)
    from_currency = state["from_currency"]
    to_currency = state["to_currency"]

    # Simulate an API call: In a real scenario, you'd make an API request to a service like {Link: Wise https://wise.com/in/currency-converter/} or {Link: Currencylayer https://currencylayer.com/}.
    # For this example, a hardcoded (simplified) rate list is used.
    rates = {
        "USD": {"EUR": 0.92, "GBP": 0.78, "INR": 83.56},
        "EUR": {"USD": 1.09, "GBP": 0.85, "INR": 91.01},
        "GBP": {"USD": 1.28, "EUR": 1.17, "INR": 107.03},
        "INR": {"USD": 0.012, "EUR": 0.011, "GBP": 0.009},
    }

    try:
        if from_currency in rates and to_currency in rates[from_currency]:
            exchange_rate = rates[from_currency][to_currency]
            state["exchange_rate"] = exchange_rate
            state["error_message"] = ""  # Clear any previous error
            print(f"  Fetched exchange rate: 1 {from_currency} = {exchange_rate} {to_currency}")
        else:
            state["exchange_rate"] = 0.0  # Indicate no rate found
            state["error_message"] = f"Exchange rate not available for {from_currency} to {to_currency}"
            print(f"  Error: {state['error_message']}")
    except Exception as e:
        state["exchange_rate"] = 0.0
        state["error_message"] = f"Error fetching exchange rate: {e}"
        print(f"  Error: {state['error_message']}")

    return state

def calculate_conversion(state: CurrencyConversionState) -> CurrencyConversionState:
    print("---Node: Calculating Conversion---")
    # This node will now execute even if there was an error in get_exchange_rate
    if state["exchange_rate"] > 0 and state["error_message"] == "":
        converted_amount = state["amount"] * state["exchange_rate"]
        state["converted_amount"] = converted_amount
        print(f"  {state['amount']} {state['from_currency']} is {converted_amount:.2f} {state['to_currency']}")
        state["structured_response"] =  ResponseFormat(status='completed', message=f"  {state['amount']} {state['from_currency']} is {converted_amount:.2f} {state['to_currency']}") 

    else:
        state["converted_amount"] = 0.0  # Still set to 0.0 if error or no rate
        print("  Skipping accurate calculation due to missing exchange rate or error from previous step.")
    return state

def final_output(state: CurrencyConversionState) -> CurrencyConversionState:
    print("---Node: Final Output---")
    if state["error_message"]:
        print(f"  Conversion failed: {state['error_message']}")
    else:
        print(f"  Final Result: {state['amount']} {state['from_currency']} = {state['converted_amount']:.2f} {state['to_currency']}")
    return state

# Construct the Graph
builder = StateGraph(CurrencyConversionState)

# Add nodes to the graph
builder.add_node("get_exchange_rate", _get_exchange_rate)
builder.add_node("calculate_conversion", calculate_conversion)
builder.add_node("final_output", final_output)

# Set the entry point
builder.set_entry_point("get_exchange_rate")

# Define direct, sequential edges
builder.add_edge("get_exchange_rate", "calculate_conversion")
builder.add_edge("calculate_conversion", "final_output")

# End the graph at the final output
builder.add_edge("final_output", END)






memory = MemorySaver()


@tool
def get_exchange_rate(
    currency_from: str = 'USD',
    currency_to: str = 'EUR',
    currency_date: str = 'latest',
):
    """Use this to get current exchange rate.

    Args:
        currency_from: The currency to convert from (e.g., "USD").
        currency_to: The currency to convert to (e.g., "EUR").
        currency_date: The date for the exchange rate or "latest". Defaults to
            "latest".

    Returns:
        A dictionary containing the exchange rate data, or an error message if
        the request fails.
    """
    try:
        response = httpx.get(
            f'https://api.frankfurter.app/{currency_date}',
            params={'from': currency_from, 'to': currency_to},
        )
        response.raise_for_status()

        data = response.json()
        if 'rates' not in data:
            return {'error': 'Invalid API response format.'}
        return data
    except httpx.HTTPError as e:
        return {'error': f'API request failed: {e}'}
    except ValueError:
        return {'error': 'Invalid JSON response from API.'}





class CurrencyAgent:
    """CurrencyAgent - a specialized assistant for currency convesions."""

    SYSTEM_INSTRUCTION = (
        'You are a specialized assistant for currency conversions. '
        "Your sole purpose is to use the 'get_exchange_rate' tool to answer questions about currency exchange rates. "
        'If the user asks about anything other than currency conversion or exchange rates, '
        'politely state that you cannot help with that topic and can only assist with currency-related queries. '
        'Do not attempt to answer unrelated questions or use tools for other purposes.'
    )

    FORMAT_INSTRUCTION = (
        'Set response status to input_required if the user needs to provide more information to complete the request.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
    )

    def __init__(self):
        # model_source = os.getenv('model_source', 'google')
        # if model_source == 'google':
        #     self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        # else:
        #     self.model = ChatOpenAI(
        #         model=os.getenv('TOOL_LLM_NAME'),
        #         openai_api_key=os.getenv('API_KEY', 'EMPTY'),
        #         openai_api_base=os.getenv('TOOL_LLM_URL'),
        #         temperature=0,
        #     )
        # self.tools = [get_exchange_rate]

        # self.graph = create_react_agent(
        #     self.model,
        #     tools=self.tools,
        #     checkpointer=memory,
        #     prompt=self.SYSTEM_INSTRUCTION,
        #     response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        # )
        self.graph  = builder.compile(checkpointer=memory)


    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        initial_request_state_1 = {"amount": 100.0, "from_currency": "USD", "to_currency": "EUR", "exchange_rate": 0.0, "converted_amount": 0.0, "error_message": "",
                                   "messages": [("user", query)],
                                   }

        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        for item in self.graph.stream(initial_request_state_1, config, stream_mode='values'):
            print("------------stream item--------")
            print(item)
            message = item['messages'][-1]
            print(message)

            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Looking up the exchange rates...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the exchange rates..',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        print("----------get_agent_response--------")
        print(config)
        current_state = self.graph.get_state(config)
        print(current_state)
        structured_response = current_state.values.get('structured_response')
        print("---------------structured_response-------")
        print(structured_response)
        # if structured_response and isinstance(
        #     structured_response, ResponseFormat
        # ):
        if structured_response:
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']