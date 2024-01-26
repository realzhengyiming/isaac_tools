import requests

if __name__ == '__main__':
    print("data spider")
    response = requests.get("https://isaac.huijiwiki.com/wiki/%E9%81%93%E5%85%B7")
    print(response.text)
