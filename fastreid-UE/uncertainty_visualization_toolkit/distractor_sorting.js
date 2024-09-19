catNum_Name_map = {
    "0": "not a distractor",
    "1": ">50% human",
    "2": "human fragment",
    "3": "object",
    "4": "background / blob",
    "5": "deferred"
}

function get_image() {
    fetch(`/get_distractor_path`)
    .then(response => response.text())
    .then(text => {
        document.getElementById("image").src = "Market-1501-v15.09.15/"+text
    })
}

get_image()
fetch(`/get_distractor_counts`).then(response => response.json()).then(json => {
    //console.log(json)
    tableData = {}
    total=0
    for (let [key, value] of Object.entries(json)) {
        tableData[catNum_Name_map[key]] = value
        total += value
      }
    tableData["total"] = total
    
    document.getElementById("stats_display").innerHTML = ""
    document.getElementById("stats_display").appendChild(createTableFromKeyValuePairs(tableData))
    get_image()
})

function createTableFromKeyValuePairs(data) {
    const table = document.createElement('table');
    const tbody = document.createElement('tbody');
  
    let rowNum = 0;
    for (const [key, value] of Object.entries(data)) {
      const row = document.createElement('tr');
  
      const numCell = document.createElement('td');
      numCell.textContent = rowNum++;
  
      const keyCell = document.createElement('td');
      keyCell.textContent = key;
  
      const valueCell = document.createElement('td');
      valueCell.textContent = value;
  
      row.appendChild(numCell);
      row.appendChild(keyCell);
      row.appendChild(valueCell);
  
      tbody.appendChild(row);
    }
  
    table.appendChild(tbody);
    return table;
  }
  

document.addEventListener('keydown', function(event) {
    // Check if the pressed key is a number between 0 and 5 on the numpad
    if (event.key >= '0' && event.key <= '5' && event.code.startsWith('Numpad')) {
        // Pass the pressed key to a function
        //console.log(event.key); // key is a string
        fetch(`/set_distractor_category?category=${encodeURIComponent(event.key)}`).then(response => response.json()).then(json => {
            //console.log(json)
            tableData = {}
            total=0
            for (let [key, value] of Object.entries(json)) {
                tableData[catNum_Name_map[key]] = value
                total += value
            }
            tableData["total"] = total
            
            document.getElementById("stats_display").innerHTML = ""
            document.getElementById("stats_display").appendChild(createTableFromKeyValuePairs(tableData))
            get_image()
        })
    }
});