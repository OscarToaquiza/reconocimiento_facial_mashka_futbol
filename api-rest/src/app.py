from flask import Flask
from config import config

#Routes
from routers import movie
from routers import entrenar
from routers import reconocer


app = Flask(__name__)

def page_not_found(error):
    return "<h1>Not found page mashka-soft 02 </h1>" , 404

if __name__ == '__main__':

    app.config.from_object(config['development'])

    #Asignar rutas
    app.register_blueprint(movie.main, url_prefix='/api/movies')
    app.register_blueprint(entrenar.main, url_prefix='/api/train')
    app.register_blueprint(reconocer.main, url_prefix='/api/recognition')

    #Manejador de error
    app.register_error_handler(404, page_not_found)

    app.run()

