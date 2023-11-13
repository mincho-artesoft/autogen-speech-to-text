from flask import Flask, render_template, request, render_template_string, send_file


app = Flask(__name__, template_folder="templates")


@app.route('/', methods=['GET'])
def index():
    return render_template('index7.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
    app.run(debug=True)
