/**
 * Show the settings (run select dialog).
 */
function showSettings() {
    document.getElementById("settings").style.display = "revert"
}

/**
 * @param {*} subclass One of "least", "mid", "most".
 * @returns An onchange event handler function for the respective input.
 */
function num_ex_input_func_factory(subclass) {
    return (e) => {
        let value = e.target.value
        document.querySelectorAll("img." + subclass + "-unc").forEach(el => {
            el.parentElement.style.display = "none"
        })
        for (let i=0; i<value; i++) {
            document.querySelectorAll("img.img-pos-" + String(i) + "." + subclass + "-unc").forEach(el => {
                el.parentElement.style.display = "inline-block"
            })
        }
    }
}


/**
 * Builds the display and sets the visibilities according to the user inputs.
 */
function render() {

    document.getElementById("settings").style.display = "none"

    // get experiments to display
    let enabled_experiments = []
    document.querySelectorAll(".experiment-select:checked").forEach((el) => {
        enabled_experiments.push(document.experiment_results[el.value])
    })

    // get column names and create columns
    column_names = []
    for (ex of enabled_experiments) {
        cols = Object.keys(ex)
        for (col of cols) {
            if (col == "name") {
                continue
            } else {
                if (!(column_names.includes(col))) {
                    column_names.push(col)
                }
            }
        }
    }
     
    // get disabled set, fun
    let disabledFunctions = document.getElementById("function-disable").value.split(",")//.map(el => "fun-"+el)
    //let disabledAggregations = document.getElementById("aggregation-disable").value.split(",")//.map(el => "agg-"+el)
    let disabledSets = document.getElementById("set-disable").value.split(",")//.map(el => "agg-"+el)
    

    diagram_el = document.getElementById("diagram")
    diagram_el.innerHTML = ""

    for (cname of column_names) {
        col = document.createElement("div")
        col.innerHTML = `
            <div class="unc-type-column" id="${cname}">
                <h3 style="f">${cname}</h3>
            </div>
        `
        diagram_el.appendChild(col)
    }

    // populate columns
    for (ex of enabled_experiments) {
        ex_name = ex.name
        for ([uncertainty_type, uncertainty_type_value] of Object.entries(ex)) {
            if (uncertainty_type == "name") {
                continue
            }

            col = document.getElementById(uncertainty_type)

            ex_block = document.createElement("div")
            ex_block.style.borderBottom = "5px solid #a00"
            ex_block.style.padding = "10px 0"

            should_color = true
            
            for ([set, set_value] of Object.entries(uncertainty_type_value)) {
                if (disabledSets.includes(set) ){
                    continue
                }
                //for ([agg, agg_value] of Object.entries(set_value)) {
                    /*if (disabledAggregations.includes(agg) ){
                        continue
                    }*/
                    for ([fun, fun_value] of Object.entries(set_value)) {
                        if (disabledFunctions.includes(fun) ){
                            continue
                        }

                        row_outer = document.createElement("div")
                        if (should_color) {
                            row_outer.classList.add("img-row-colored")
                        }
                        should_color = !should_color
                        row_title = document.createElement("div")
                        row_title.style.padding = "5px 12px 0 12px"
                        row_title.innerHTML = `${ex_name}: ${set}/${fun}`
                        row_outer.appendChild(row_title)

                        row = document.createElement("div")
                        row.classList.add("f-" + fun)
                        //row.classList.add("a-" + agg)
                        row.classList.add("s-" + set)
                        row.classList.add("img-row")
                        row.style.display = "flex"
                        row.style.flexDirection = "row"

                        fun_value["least_uncertain"].forEach( (path, index) => {
                            div = document.createElement("div")
                            img = document.createElement("img")
                            img.src = market_path + "/" + path.split("/").slice(-2).join("/")
                            img.classList.add("img-pos-" + String(index), "least-unc") 
                            
                            div.appendChild(img)
                            
                            overlay = document.createElement("div")
                            overlay.innerHTML = fun_value["least_uncertain_scores"][index].toExponential(2) // TODO: visibility must be changed on div instead of img, round score -> 1.23e-3
                            overlay.style.position = "absolute"
                            overlay.style.color = "white"
                            overlay.style.transform = "translate(0.25rem,-1.5rem)"
                            overlay.style.textShadow = "-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000"


                            div.appendChild(overlay)
                            row.appendChild(div)
                        })

                        let divider = document.createElement("div")
                        divider.classList.add("ranking-divider")
                        row.appendChild(divider)

                        fun_value["medium_uncertain"].forEach( (path, index) => {
                            div = document.createElement("div")
                            img = document.createElement("img")
                            img.src = market_path + "/" + path.split("/").slice(-2).join("/")
                            img.classList.add("img-pos-" + String(index), "mid-unc")
                            div.appendChild(img)

                            overlay = document.createElement("div")
                            overlay.innerHTML = fun_value["medium_uncertain_scores"][index].toExponential(2)
                            overlay.style.position = "absolute"
                            overlay.style.color = "white"
                            overlay.style.transform = "translate(0.25rem,-1.5rem)"
                            overlay.style.textShadow = "-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000"


                            div.appendChild(overlay)

                            row.appendChild(div)
                        })
                        
                        let divider2 = document.createElement("div")
                        divider2.classList.add("ranking-divider")
                        row.appendChild(divider2)

                        fun_value["most_uncertain"].forEach( (path, index) => {
                            div = document.createElement("div")
                            img = document.createElement("img")
                            img.src = market_path + "/" + path.split("/").slice(-2).join("/")
                            img.classList.add("img-pos-" + String(fun_value["most_uncertain"].length - index - 1), "most-unc")

                            div.appendChild(img)

                            overlay = document.createElement("div")
                            overlay.innerHTML = fun_value["most_uncertain_scores"][index].toExponential(2) // TODO: visibility must be changed on div instead of img, round score -> 1.23e-3. done?
                            overlay.style.position = "absolute"
                            overlay.style.color = "white"
                            overlay.style.transform = "translate(0.25rem,-1.5rem)"
                            overlay.style.textShadow = "-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000"

                            // TODO: make entire columns hide-able (e.g. dont care about MODEL)

                            // TODO: wenn alle bilder eines kleineren sets visualisiert werden: klick auf ein Bild -> Popup: Position auf allen (relevanten) Skalen

                            div.appendChild(overlay)

                            row.appendChild(div)
                        })
                        
                        row_outer.appendChild(row)
                        ex_block.appendChild(row_outer)
                    }
                //}
            }

            col.appendChild(ex_block)
            

        }
    }

    // TODO: is this slow? to do it every time? might be able to increase performance.

    // Create a new 'change' event to make sure that the proper number of images is displayed
    var event = new Event('change');

    // Dispatch it.
    for (subclass of ["least", "mid", "most"]) {
        document.getElementById("num-ex-input-" + subclass).dispatchEvent(event);
    }
}
document.render=render // register here for easy accessibility


/**
 * Adds the entries to the settings / runs selection dialog for each run as well as grouping them.
 */
function populate_settings(){
    settings_div = document.getElementById("settings")
    let settings_pane = document.createElement("div")
    settings_pane.id = "settings-pane"

    groups = {}

    for ([i, result] of document.experiment_results.entries()) {

        if (result.name.split("/")[result.name.split("/").length-1] == "test") {
            continue // could add config option for displaying these but essentially, why would you want to look at incomplete runs?
        }

        if (result.name.includes("/")){
            if (groups[result.name.split("/")[0]] == null) {

                // greate group (name = just first part)
                let experiment_group = document.createElement("div")
                let group_label = document.createElement("div")
                group_label.innerHTML = `
                    <div style="display:flex; flex-direction: row; background-color:transparent;" class="experiment-select-group-label">
                        <input type="checkbox" value="${Object.keys(groups).length}">
                        <div class="experiment-select-group-label-text-parent" style="display:flex;width:100%;">
                            <div class="experiment-select-group-label-text">
                                ${result.name.split("/")[0]}
                            </div>
                            <div style="margin: 0 0 0 auto;">▾</div>
                        </div>
                    </div>
                ` 
                experiment_group.appendChild(group_label)

                group_label.querySelector("input").onclick = (e) => {
                    group = experiment_group

                    group.querySelectorAll(".experiment-select-row input").forEach((entry_el) => {
                        entry_el.checked = e.target.checked
                    })
                }

                experiment_group.querySelector(".experiment-select-group-label-text-parent").onclick = (e) => {
                    group = experiment_group

                    group.querySelectorAll(".experiment-select-row").forEach((el) => {
                        // toggle visibility and indent
                        el.style.display = el.style.display == "none" ? "flex" : "none"
                        el.style.paddingLeft = el.style.paddingLeft == "1rem" ? "0" : "1rem"
                    })

                    label = group.querySelector(".experiment-select-group-label")

                    // toggle color
                    label.style.backgroundColor = label.style.backgroundColor == "transparent" ? "#777" : "transparent"
                }

                
                settings_pane.appendChild(experiment_group)

                groups[result.name.split("/")[0]] = experiment_group
            } 

            experiment_group = groups[result.name.split("/")[0]]

            // create single entries
            let ex_entry = document.createElement("div")
            ex_entry.innerHTML = `
                <div style="display:none; flex-direction: row; padding: 0;" class="experiment-select-row">
                    <input type="checkbox" value="${i}" class="experiment-select">${result.name}
                </div>
            `  
            experiment_group.appendChild(ex_entry)

            // attach single entries to group
            // indent single entries
            // make collapsible
            // have color information for toggle status

        } else {
            let div = document.createElement("div")
            div.innerHTML = `
                <div style="display:flex; flex-direction: row">
                    <input type="checkbox" value="${i}" class="experiment-select">${result.name}
                </div>
            `    
            settings_pane.appendChild(div)
        }
    }

    render_button = document.createElement("button")
    render_button.innerHTML = "Render"
    render_button.onclick = render
    select_all_button = document.createElement("button")
    select_all_button.innerHTML = "Select All"
    select_all_button.onclick = (e) => {
        document.querySelectorAll(".experiment-select").forEach(el => {
            el.checked = true
        })
    }
    settings_div.appendChild(settings_pane)
    settings_div.appendChild(render_button)
    settings_div.appendChild(select_all_button)
}   


// #################################################################################################


// Add onchange functions to inputs for number of images
for (subclass of ["least", "mid", "most"]) {
    document.getElementById("num-ex-input-" + subclass).onchange = num_ex_input_func_factory(subclass)
}

//TODO: pin row (so it stays when selecting different runs/whatever, to enable even more flexibility in comparing them)

// read config
//CONFIG
//console.log(CONFIG)
market_path = CONFIG["market_path"]
results_path = CONFIG["results_path"]

document.experiment_results = []

// read uncertain_images files and parse them, then set up and render the visualizer
fetch(`/get_uncertain_images_files?path=${encodeURIComponent(results_path)}`)
.then(response => response.text())
.then(text => {

    // need to replace the `Infinity` and `NaN` values created by python into JS-understandable format.
    document.experiment_results = (JSON.parse(text.replaceAll("Infinity", "\"Infinity\"").replaceAll("-\"Infinity\"", "\"-Infinity\"").replaceAll("NaN", "null")));

    populate_settings()

    // render is caused by user input
});