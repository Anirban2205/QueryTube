from app import Query_model

if __name__ == "__main__":
    url = input("Enter the video link: ")
    while True:
        query = input("Enter the query: ")
        answer = Query_model(url).get_answer(query)
        print(answer)
        