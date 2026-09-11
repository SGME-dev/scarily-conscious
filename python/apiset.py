from caveclient import CAVEclient

client = CAVEclient()

# Add overwrite=True to replace the old stored key
client.auth.save_token(token="d77af9021e9f3e623bb372c363a49429", overwrite=True)
print("Token saved successfully!")

# Test connection
try:
    client = CAVEclient('flywire_fafb_public')
    print("Connected successfully to flywire_fafb_public!")
except Exception as e:
    print("Error connecting:", e)