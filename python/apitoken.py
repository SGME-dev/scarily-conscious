from caveclient import CAVEclient

client = CAVEclient('flywire_fafb_production')
print("Connected to dataset:", client.datastack_name)