# from flask import Flask, request, jsonify
# import pandas as pd
# from src.nlp_model import get_query_embedding, find_top_matches
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)
# # Load dataset
# dataset_path = "../datasets/processed_data.csv"
# data = pd.read_csv(dataset_path)
# data['embeddings'] = data['embeddings'].apply(eval)  # Convert string to list

# @app.route('/search', methods=['POST'])
# def search():
#     query = request.json.get('query', '')
#     if not query:
#         return jsonify({"error": "Query cannot be empty"}), 400
    
#     query_embedding = get_query_embedding(query)
#     results = find_top_matches(query_embedding, data)
    
#     response = [
#         {"rank": i + 1, "link": row['link'], "summary": row['summary'], "similarity": row['similarity']}
#         for i, row in results.iterrows()
#     ]
#     return jsonify(response)

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, jsonify
import pandas as pd
from src.nlp_model import get_query_embedding, find_top_matches
from flask_cors import CORS  # Importing CORS to allow cross-origin requests

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

# Load dataset
dataset_path = "../datasets/processed_data.csv"
data = pd.read_csv(dataset_path)
data['embeddings'] = data['embeddings'].apply(eval)  # Convert string to list

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    query_embedding = get_query_embedding(query)  # Get the query embedding using your model
    results = find_top_matches(query_embedding, data)  # Find top matches based on embeddings
    
    # Assuming results is a DataFrame and contains columns like 'link', 'summary', and 'similarity'
    response = [
        {"rank": i + 1, "link": row['link'], "summary": row['summary'], "similarity": row['similarity']}
        for i, row in results.iterrows()
    ]
    
    return jsonify(response)  # Returning the list of results to the frontend

if __name__ == "__main__":
    app.run(debug=True)
