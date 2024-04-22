window.addEventListener('DOMContentLoaded', (event) => {
    const form = document.getElementById('uploadForm');
    const resultContainer = document.getElementById('result');
    const loading = document.getElementById('loading');
    const prediction = document.getElementById('prediction');
    const imagePreview = document.getElementById('imagePreview');

    form.addEventListener('submit', (event) => {
        event.preventDefault();
        loading.style.display = 'block';
        prediction.textContent = '';
        imagePreview.innerHTML = '';

        const fileInput = document.getElementById('fileInput');
        const file = fileInput.files[0];

        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loading.style.display = 'none';
            prediction.textContent = `Classification Result: ${data.result}`;

            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.alt = 'Uploaded Image';
            img.classList.add('result-image');
            imagePreview.appendChild(img);
        })
        .catch(error => {
            loading.style.display = 'none';
            prediction.textContent = 'Error occurred during prediction.';
            console.error('Error:', error);
        });
});


