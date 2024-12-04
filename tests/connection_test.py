import asyncio
from deriv_api import DerivAPI

async def main():
    """
    A simple script to test the connection to the Deriv API.
    """
    # Replace with your actual app_id and API token
    app_id = '66244'        # e.g., '1089'
    api_token = 'QnW0uZaQXJnpMZ3'  # Replace with your Deriv API token

    # Initialize the DerivAPI instance
    api = DerivAPI(app_id=app_id)
    print("Deriv API instance created.")

    try:
        # Authorize with API token
        authorize_response = await api.authorize(api_token)
        if 'error' in authorize_response:
            print(f"Authorization failed: {authorize_response['error']['message']}")
        else:
            print("Authorization successful.")

            # Send a ping request
            ping_response = await api.ping({'ping': 1})
            if 'ping' in ping_response:
                print("Ping successful. Connection is working.")
            else:
                print(f"Ping failed: {ping_response}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await api.disconnect()
        print("Disconnected from Deriv API.")

if __name__ == "__main__":
    asyncio.run(main())
