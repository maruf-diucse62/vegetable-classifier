function uploadImage() {
    let file = document.getElementById('fileInput').files[0];
    if (!file) {
        alert('Please select an image');
        return;
    }

    let preview = document.getElementById('preview');
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';

    let formData = new FormData();
    formData.append('image', file);

    fetch('https://vegetable-classifier-lhm2.onrender.com/predict', {
    method: 'POST',
    body: formData
    })
    .then(res => res.json())
    .then(data => {
        // Use template literals for dynamic values
        document.getElementById('result').innerHTML = `
            <h3>Prediction: ${data.label}</h3>
            <p>Confidence: ${data.confidence}</p>
        `;
    })
    .catch(err => {
        console.error(err);
        alert('Error connecting to the backend!');
    });
}
