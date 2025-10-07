from fastapi import FastAPI 

app=FastAPI() # http://localhost:8000/ 
@app.get("/") # Route the url 
def welcome(): 
return {"message": "Hello, World!"}