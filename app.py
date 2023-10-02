from flask import Flask, render_template, request, send_file
from model import run_style_transfer
from PIL import Image


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_image = request.files['input-image']
        style_image = request.files['style-image']
        num_epochs = int(request.form['num-epochs'])
        model = request.form.get('model')
        
        best, best_loss = run_style_transfer(
            input_image, 
            style_image, 
            num_iterations=num_epochs,
            model=model)
        img = Image.fromarray(best) 
        img.save('templates/output.png')
    
    
        
        return render_template('index.html', output_image=True)
    else:
        return render_template('index.html')

@app.route('/output.png')
def output_image():
    return send_file('templates/output.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
