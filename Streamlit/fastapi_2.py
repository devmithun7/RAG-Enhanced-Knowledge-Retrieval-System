import snowflake.connector
import warnings
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException


warnings.filterwarnings('ignore')

class Query(BaseModel):
    sql_query: str

app= FastAPI()


# Snowflake connection parameters

Snowflake_user= "anee13",
Snowflake_password= "AK@indinc101010",
Snowflake_account= "DPEDYNZ.SR19718",
Snowflake_warehouse= "cfa_dev_warehouse",
Snowflake_database= "DETAILS",
Snowflake_schema= "PUBLIC"

@app.post('/query-snowflake')
def query_snowflake(query: Query):
    conn = None  
    cursor = None
    try:
        conn: SnowflakeConnection = snowflake.connector.connect(
            user= Snowflake_user,
            password= Snowflake_password,
            account= Snowflake_account,
            warehouse=Snowflake_warehouse,
            database= Snowflake_database,
            schema= Snowflake_schema
            )
        cursor= conn.cursor()
        cursor.execute(query.sql_query)
        result= cursor.fetchall()
        return {'data': result}
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e)) 
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()