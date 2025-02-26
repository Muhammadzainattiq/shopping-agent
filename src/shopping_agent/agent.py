from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage
from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import asyncio

load_dotenv()

# model = ChatOpenAI(model="gpt-4o-mini")
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


mock_data = {
    "laptop": [
        {
            "name": "Laptop A",
            "price": "$999",
            "url": "http://example.com/laptop-a",
            "description": "A powerful laptop with 16GB RAM and 512GB SSD."
        },
        {
            "name": "Laptop B",
            "price": "$1299",
            "url": "http://example.com/laptop-b",
            "description": "A lightweight laptop with a stunning display and long battery life."
        },
    ],
    "phone": [
        {
            "name": "Phone A",
            "price": "$699",
            "url": "http://example.com/phone-a",
            "description": "A smartphone with a great camera and fast performance."
        },
        {
            "name": "Phone B",
            "price": "$899",
            "url": "http://example.com/phone-b",
            "description": "A premium smartphone with a sleek design and advanced features."
        },
    ],
}

@tool
def show_all_products():
    """Retrieve and display all available products.

    Returns:
        list: A list of all available products with their details.
    """
    return mock_data

@tool
def search_product(query: str):
    """Search for products by category.

    Args:
        query (str): The category name to search for.

    Returns:
        list: A list of products matching the specified category.
              Returns an empty list if no products are found.
    """
    return mock_data.get(query.lower(), [])

@tool
def add_to_cart(product_name: str):
    """Add a specified product to the shopping cart.

    Args:
        product_name (str): The name of the product to add.

    Returns:
        str: A confirmation message if the product is added successfully,
             or an error message if the product is not found.
    """
    for category, products in mock_data.items():
        for product in products:
            if product['name'].lower() == product_name.lower():
                cart.append(product)
                return f"{product_name} has been added to your cart."
    return f"{product_name} not found."

@tool
def see_cart():
    """Retrieve the current items in the shopping cart.

    Returns:
        list or str: A list of items in the cart if it's not empty,
                     otherwise a message indicating the cart is empty.
    """
    if not cart:
        return "Your cart is empty."
    return cart

@tool
def remove_from_cart(product_name: str):
    """Remove a specified product from the shopping cart.

    Args:
        product_name (str): The name of the product to remove.

    Returns:
        str: A confirmation message indicating whether the product was removed.
    """
    global cart
    cart = [product for product in cart if product['name'].lower() != product_name.lower()]
    return f"{product_name} has been removed from your cart."

@tool
def checkout(address: str, phone_no: int, card_no: int):
    """Process the checkout by finalizing the order and clearing the cart.

    Args:
        address (str): The delivery address.
        phone_no (int): The customer's phone number.
        card_no (int): The payment card number.

    Returns:
        str: A confirmation message with the total price and delivery details,
             or an error message if the cart is empty.
    """
    if not cart:
        return "Your cart is empty. Nothing to checkout."
    total_price = sum(float(product['price'].replace('$', '')) for product in cart)
    cart.clear()  # Clear the cart after checkout
    return f"Checkout complete! Your total is ${total_price:.2f}. You will receive your order in 3 working days."


tools = [show_all_products, search_product, add_to_cart, see_cart, remove_from_cart, checkout]



tools_by_name = {tool.name: tool for tool in tools}
sys_msg = """You are a shopping agent which help users do shopping.Introduce your self as shopping assistant and ask them what help they require. Users can search for products,
 add them to cart, remove things from cart , see the cart, and can also checkout. You will show them the products and then ask them if they want it to cart. When they add it to cart,
 ask them if they continue shopping or checkout."""

@task
def call_model(messages):
    """Call model with a sequence of messages."""
    response = model.bind_tools(tools).invoke([sys_msg] + messages)
    return response


@task
def call_tool(tool_call):
    tool = tools_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    return ToolMessage(content=observation, tool_call_id=tool_call["id"])


checkpointer = MemorySaver()


@entrypoint(checkpointer=checkpointer)
def agent(messages, previous):
    if previous is not None:
        messages = add_messages(previous, messages)

    llm_response = call_model(messages).result()
    while True:
        if not llm_response.tool_calls:
            break

        # Execute tools
        tool_result_futures = [
            call_tool(tool_call) for tool_call in llm_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]

        # Append to message list
        messages = add_messages(messages, [llm_response, *tool_results])

        # Call model again
        llm_response = call_model(messages).result()

    # Generate final response
    messages = add_messages(messages, llm_response)
    return entrypoint.final(value=llm_response, save=messages)


config = {"configurable": {"thread_id": "1"}}

async def conversation_loop():
    while True:
        inp = input("\nEnter (or type 'exit' to quit): ")
        if inp.lower() == "exit":
            print("Exiting the shopping assistant. Have a great day!")
            break

        user_message = {"role": "user", "content": f"{inp}"}
        print("\nUser:", inp)

        for step in agent.stream([user_message], config):
            for task_name, message in step.items():
                if task_name == "agent":
                    continue  # Just print task updates
                print(f"\n{task_name}:")
                message.pretty_print()


# Run the agent and stream the messages to the console.
async def main() -> None:
    await conversation_loop()


def run_main():
    asyncio.run(main())

if __name__ == "__main__":
    run_main()