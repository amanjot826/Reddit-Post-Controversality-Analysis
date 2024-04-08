function analyzePost() {
    const postBody = document.getElementById('postBody').value;
    // Perform analysis on the post body
    // ...
    // Display the analysis result
    displayResult('Post Analysis', result);
}

function analyzeComment() {
    const commentText = document.getElementById('commentInput').value;
    // Perform analysis on the comment text
    // ...
    // Display the analysis result
    displayResult('Comment Analysis', result);
}

function analyzeImage() {
    const imageInput = document.getElementById('imageInput');
    const file = imageInput.files[0];

    if (file) {
        // Perform analysis on the image file
        // ...
        // Display the analysis result
        displayResult('Image Analysis', result);
    } else {
        displayResult('Image Analysis', 'Please select an image file.');
    }
}

function displayResult(title, result) {
    const analysisResult = document.getElementById('analysisResult');
    analysisResult.innerHTML = `<h2>${title}</h2><p>${result}</p>`;
}
function displayResult(isControversial) {
    var resultDiv = document.getElementById("analysisResult");
    resultDiv.textContent = isControversial ? "This content is controversial." : "This content is not controversial.";
}
