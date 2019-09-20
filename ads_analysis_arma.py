import sys
import arma
import datetime

if __name__ == "__main__":
    

    def validate(date_text):
        try:
            datetime.datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")
        
    validate(sys.argv[2])
    arma.predict_arma(sys.argv[1],sys.argv[2])
