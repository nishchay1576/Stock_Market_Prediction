from flask import Flask, render_template
import pickle
import pandas as pd
import datetime

app = Flask(__name__,
            static_url_path='',
            static_folder = '/Users/nishchay/Desktop/Sentimental-Analysis-Flask-Heroku-main/Sentimental-Analysis-Flask-Heroku-main/templates/static',
            template_folder='/Users/nishchay/Desktop/Sentimental-Analysis-Flask-Heroku-main/Sentimental-Analysis-Flask-Heroku-main/templates')
model = pickle.load(open('model.pkl', 'rb'))
modelvect = pickle.load(open('modelvect.pkl','rb'))

data = pd.read_csv('mycsv.csv', encoding = "ISO-8859-1")
test=data.iloc[1:,0:]


testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
basictest = modelvect.transform(testheadlines)
predictions =model.predict(basictest)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictions',methods=['POST'])
def predict():
    output = model.predict(basictest[-1])
  
    tomorrow= datetime.date.today() + datetime.timedelta(days=1)
   
    if output == 1:        
        return render_template('index.html', prediction_text='The Prices will Move Upwards on ${}'.format(tomorrow))
    else:
        return render_template('index.html', prediction_text='The Prices will Not(X) Move Upwards on {}'.format(tomorrow))


if __name__ == "__main__":
    app.run(debug=True)
   
'''    
   x= datetime.date.today()
    print(x)
x= datetime.datetime.now()
print(x.year)
'''