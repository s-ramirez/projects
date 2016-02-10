from flask_restful import Resource, reqparse
from models.application import Application
from data.application import ApplicationDB

class Applications(Resource):
    def get(self):
        return {'status': 'success'}
    def post(self):
        appDB = ApplicationDB()
        try:
            #Parse the arguments
            parser = reqparse.RequestParser()
            parser.add_argument('title', type=str, help='Title of the application')
            parser.add_argument('url', type=str, help='URL of the application')
            parser.add_argument('imageUrl', type=str, help='URL of the application\'s icon')
            args = parser.parse_args()

            app = Application(args['title'], args['url'], args['imageUrl'])
            data = appDB.create(app)

            if data:
                return {'status': 'success'}
            else:
                return {'status': 'error'}
        except Exception as e:
            return {'error': str(e)}
