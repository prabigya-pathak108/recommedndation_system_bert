from routes import api_routes


def register_routes(app):
    app.include_router(
        router=api_routes.router,
        prefix='/api',
        responses={404: {'description': 'Not found'}},
    )



