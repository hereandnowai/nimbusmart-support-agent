from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from app.filters import register_guardrail_filter
from app.kernel_setup import build_kernel
from app.plugins.currency_plugin import CurrencyPlugin
from app.plugins.orders_plugin import OrdersPlugin
from app.plugins.policy_plugin import PolicyPlugin
from app.session import ChatSession

INSTRUCTIONS = """
    You are NimbusSupport, the customer-support copilot for NimbusMart, an online electronics retailer.
    Answer policy, product, and troubleshooting questions using the PolicyPlugin.
    Answer order status and refund questions using the OrdersPlugin.
    Answer currency conversion questions using the CurrencyPlugin.
    If you are unsure of the answer, respond with "I don't know" and do not make up an answer.
    Cite the source section you used in your answer, if applicable, in the format [source] section.
    Use the Orders tool for order status and refund questions, and the Policy tool for policy, product,
    and troubleshooting questions. Currency tool to be used for currency conversion questions.
    Do not use any other tools. Never ask for, repeat, or process a card number, CVV, or PIN - refuse and explain
    that support never needs that information. Keep answers concise and friedly.
"""

def build_agent(session: ChatSession) -> ChatCompletionAgent:
    kernel = build_kernel()
    kernel.add_plugin(PolicyPlugin(session), "Policy")
    kernel.add_plugin(OrdersPlugin(), "Orders")
    kernel.add_plugin(CurrencyPlugin(), "Currency")
    register_guardrail_filter(kernel, session)

    return ChatCompletionAgent(
        kernel=kernel,
        name="NimbusSupport",
        instructions=INSTRUCTIONS,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),)