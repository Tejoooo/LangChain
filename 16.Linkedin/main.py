import requests
API_KEY = "GP90CyHPUHJsSZPXr2XTfQ"

headers = {'Authorization': 'Bearer ' + API_KEY}
api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
linkedin_profile_url = 'https://www.linkedin.com/in/tejo-kaushal-4a32b5256/'

response = requests.get(api_endpoint,
                        params={'url': linkedin_profile_url},
                        headers=headers)

print(response.text)