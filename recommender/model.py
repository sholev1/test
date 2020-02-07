from flask import Flask
from flask import request
from flask import jsonify
from recommender import recommender

app = Flask(__name__)


@app.route('/api/movie_recommendations/<int:id>_<int:n_recommendations>', methods=['GET'])
def get_recommendations(id, n_recommendations):

    # query = str(request.args['Query'])
    # return query
    d = {}
    # d['Query'] = str(request.args['Query'])
    rec = recommender()

    d['category'] = rec.make_recommendations(id, n_recommendations)
    print(d['category'])
    return jsonify(d)
    #
    # @app.route('/api', methods=['POST'])
    # def hello_world():
    #     d = {}
    #     d['Query'] = str(request.args['Post'])
    #     return jsonify(d)


app.run()
