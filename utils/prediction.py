from google.oauth2 import service_account
from google.auth.transport.requests import Request

creds = service_account.Credentials.from_service_account_file(
    "credentials/google_Api.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
creds.refresh(Request())  # Force token refresh

print(f"""
✅ Service Account: {creds.service_account_email}
✅ Token Expires: {creds.expiry}
✅ Project ID: {creds.project_id}
""")