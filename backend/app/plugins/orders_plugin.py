from semantic_kernel.functions import kernel_function
from app.orders_data import ORDERS

class OrdersPlugin:
    @kernel_function(name="get_order_status", description="Get the status of an order by its ID.")
    def status(self, order_id: str) -> str:
        order = ORDERS.get(order_id.upper())
        if not order:
            return "not found"
        tracking = order["tracking"] or "not yet available"
        return (
            f"status={order['status']};, items={', '.join(order['items'])};"
            f"placed_on={order['placed_on']};, tracking={tracking}"
        )
    
    @kernel_function(description = "Get the refund amount owed on an order, in US dollars (0 if non is due).")
    def refund_amount(self, order_id: str) -> float:
        order = ORDERS.get(order_id.upper())
        return order["refund_due_usd"] if order else 0.0