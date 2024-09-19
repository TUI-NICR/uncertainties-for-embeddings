function render() {
    // gather user inputs
    num_bins = document.getElementById("num-bins").value
    set = document.getElementById("set").value
    uncertainty_type = document.getElementById("unc-type").value
    window_lower = document.getElementById("window-lo").value
    window_upper = document.getElementById("window-hi").value
    score_function = document.getElementById("score-func").value
    density = Number(document.getElementById("density").checked)

    if (!document.getElementById("window-enable").checked) {
        window_lower = window_upper = ""
    }

    if (window_lower == "") {
        window_lower = "None"
    }
    if (window_upper == "") {
        window_upper = "None"
    }

    // send to server
    // server computes image and saves it.
    fetch(`/render_score_hist?bins=${num_bins}&set=${set}&type=${uncertainty_type}&low=${window_lower}&high=${window_upper}&func=${score_function}&density=${density}`).then(response => {
        // update img URI to refresh image 
        document.getElementById("hist").src = `score_histogram.png?t=${Date.now()}`
    })
}

document.render = render