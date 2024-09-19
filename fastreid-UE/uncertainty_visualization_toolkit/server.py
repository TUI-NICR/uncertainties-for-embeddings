from http.server import SimpleHTTPRequestHandler, HTTPServer
import os, json
import random
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from urllib.parse import urlparse, parse_qs

SERVER_PORT = 8000

# path to Market dataset folder
MARKET_PATH = "./Market-1501-v15.09.15"

# get the filenames of all the distractor images from the gallery set of Market
DISTRACTOR_PATHS = [el for el in filter(lambda filename: filename.startswith("0000"), os.listdir(os.path.join(MARKET_PATH, 'bounding_box_test')))]


# helper functions

def json_load_progress_bar_hook(obj):
    """
    A hook to pass to json.load to get a progress bar for loading the file.

    NOTE: There is still a substantial delay between json.load call and appearance of the progress bar.
    """
    for key, value in obj.items():
        # don't care about these keys
        if key in ["mean_vector", "variance_vector", "variance_of_mean_vector", "variance_of_variance_vector"] or key[-4:] == ".jpg":
            continue
        
        if type(value) is dict:
            pbar = tqdm(value.keys())
            for _ in pbar:
                pbar.set_description("Loading " + str(key))
    return obj

def search_unc_imgs_rec(path):
    """
    Traverses the file tree starting at `path` and returns a list of paths to all files named `"uncertain_images.json"` in it.
    """
    ret = []

    # list all files in dir
    for filename in os.listdir(path):
        new_path = os.path.join(path, filename)
        if filename == "uncertain_images.json":
            ret.append(new_path)
        elif os.path.isdir(new_path):
            # recursion
            ret = ret + search_unc_imgs_rec(new_path)
    return ret

def get_random_distractor_path():
    """
    Returns a string containing the path inside the Market dataset for a randomly chosen distractor.
    """
    return os.path.join("bounding_box_test", random.choice(DISTRACTOR_PATHS))
    

# DC = distractor classification
DC_active_path = ""
DC_classified_images = None
DC_num_per_category = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0
}
raw_model_outputs = None

class CustomHandler(SimpleHTTPRequestHandler):

    def __init__(self, request, client_address, server, *, directory=None) -> None:
        
        super().__init__(request, client_address, server, directory=directory)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def respond(self, response: str):
        """
        Basic success response.
        """
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(bytes(response, 'utf-8'))

    def do_GET(self):
        """
        Main handler function.
        """
        
        parsed_path = urlparse(self.path)
        params = parse_qs(parsed_path.query)

        global DC_active_path

        if parsed_path.path == '/get_uncertain_images_files':
            files_json = self.uncertainty_ranking_visualizer(params)
            self.respond(files_json)

        elif parsed_path.path == "/get_distractor_path":
            path = self.get_distractor_path()
            self.respond(path)

        elif parsed_path.path == "/set_distractor_category" and DC_active_path != "":
            num_per_category_json = self.set_distractor_category(params)
            self.respond(num_per_category_json)
            
        elif parsed_path.path == "/get_distractor_counts":
            global DC_num_per_category
            self.respond(json.dumps(DC_num_per_category))

        elif parsed_path.path == "/render_score_hist":
            self.render_score_histogram(params)
            self.respond("OK")

        else:
            super().do_GET()
    
    def uncertainty_ranking_visualizer(self, params):
        """
        Searches for all files in the folder given by the query parameter `path`, as well as any
        files called `uncertain_images.json` in the filetree starting at `path` and loads them as JSON
        objects.

        Returns the objects collected in a list as JSON text.

        This is to load the relevant data for the uncertainty ranking visualizer.
        """

        path = ""

        # Check if 'param_name' exists in the query parameters
        if 'path' in params:
            # Access the value of 'param_name'
            path = params['path'][0]
            # Now you can use param_value in your Python code

        files = os.listdir(path)
        files_json = []
        for file in tqdm(files):

            if os.path.isfile(os.path.join(path, file)):
                # TODO: could add filter to only parse JSON files
                with open(os.path.join(path, file), "r") as f:
                    files_json.append(json.load(f))
            else:
                # recursive search for uncertain_images.json
                ui_paths = search_unc_imgs_rec(os.path.join(path, file))
                for ui_path in ui_paths:
                    with open(ui_path, "r") as f:
                        files_json.append(json.load(f))

        return json.dumps(files_json)

    def get_distractor_path(self):
        """
        Returns a new (= not classified before) random path to a distractor.
        """

        # on first call, load the save-state file 
        global DC_classified_images
        global DC_num_per_category
        if DC_classified_images == None:
            DC_classified_images = []
            # Open the file
            with open('distraction_levels.txt', 'r') as file:
                # Iterate over each line
                for line in file:
                    # Split the line at the comma
                    path, category = line.strip().split(',')

                    DC_classified_images.append(path)
                    DC_num_per_category[category] += 1

        path = ""
        condition = len(DC_classified_images) < len(DISTRACTOR_PATHS)
        while condition:
            path = get_random_distractor_path()
            condition = path in DC_classified_images

        global DC_active_path
        DC_active_path = path

        return path
    
    def set_distractor_category(self, params):
        """
        Sets the category for the "currently active" (previously requested) distractor (`DC_active_path`).
        The category is given by the query parameter `category`.

        Returns the numbers of categorized distractors in each category as a JSON string.
        """
        category = ""
        if "category" in params:
            category = params["category"][0]

        # add active path to classified images
        global DC_active_path
        global DC_classified_images
        global DC_num_per_category
        DC_classified_images.append(DC_active_path)

        # add active path to file with category
        with open('distraction_levels.txt', 'a') as file:
            line = f"{DC_active_path},{category}"
            file.write(line + '\n')

        # increment number in that category
        DC_num_per_category[category] += 1
        DC_active_path = ""

        return json.dumps(DC_num_per_category)
    
    def get_RMO_vectors(self, set_id: str, vector_type: str):
        """
        Returns the vectors of raw model outputs (RMO) as specified for the given subset of images and type of vector.
        """
        global raw_model_outputs
        if vector_type == "DD":
            data = np.array(self.get_RMO_vectors(set_id, "variance_vector"))
            dist = np.array(self.get_RMO_vectors(set_id, "variance_of_variance_vector"))
            return data / dist
        return [raw_model_outputs["data"][name][vector_type] for name in raw_model_outputs["sets"][set_id]]
    
    def render_score_histogram(self, params):
        """
        Calculates score values for the specified raw vectors and renders a histogram of their distribution as a png file.
        """
        # get parameters
        num_bins = int(params['bins'][0])
        set_id: str = params['set'][0] # can be multiple (comma-separated) -> secondaries are overlayed transparently
        window_lower: str = params['low'][0]
        window_upper: str = params['high'][0]
        uncertainty_type: str = params["type"][0]
        score_function_id: str = params['func'][0]
        density = bool(int(params["density"][0]))

        additional_set_ids = None

        if len(set_id.split(",")) > 1:
            additional_set_ids = set_id.split(",")
            set_id = additional_set_ids.pop(0)

        global raw_model_outputs
        # load raw data file if not already loaded
        if raw_model_outputs == None:
            with open("raw_model_outputs.json", "r") as f:
                print("Loading raw_model_outputs.json...")
                raw_model_outputs = json.load(f, object_hook=json_load_progress_bar_hook)
                print("Done!")

        # select relevant vectors
        vector_type_map = {
            "data": "variance_vector",
            "model": "variance_of_mean_vector",
            "dist": "variance_of_variance_vector",
            "DD": "DD"
        }
        data = self.get_RMO_vectors(set_id, vector_type_map[uncertainty_type])
        # data is [[1, 2, 3, ..., 2048], ...]

        # compute chosen score function
        score_function_map = {
            "L1": lambda x: np.linalg.norm(x, 1),
            "L2": lambda x: np.linalg.norm(x, 2),
            "max": lambda x: np.linalg.norm(x, np.inf),
            "min": lambda x: np.linalg.norm(x, -np.inf),
            "avg": lambda x: np.mean(x),
            "entropy": lambda x: 0.5*np.sum(np.log(x)) + 1.4189385332 * len(x) # diagonal covariance of multivariate normal
        }

        scores = [x for x in map(score_function_map[score_function_id], data)]

        # set histogram range
        window_lower = min(scores) if window_lower == "None" else float(window_lower)
        window_upper = max(scores) if window_upper == "None" else float(window_upper)

        # safety net
        if not window_lower < window_upper:
            window_lower = min(scores)
            window_upper = max(scores)

        # create and save histogram
        plt.figure(figsize=(16,9))
        bin_values, bins, patches = plt.hist(scores, bins=num_bins, range=(window_lower, window_upper), density=density, label=set_id)
        

        mean = np.mean(scores)
        std = np.std(scores)
        
        plt.errorbar(mean, max(bin_values) * 1.02, xerr=std, color=patches[0].get_facecolor(), fmt="|") # also plot mean/std slightly above highest bin

        if additional_set_ids != None:
            for add_set_id in additional_set_ids:
                add_scores = [x for x in map(score_function_map[score_function_id], self.get_RMO_vectors(add_set_id, vector_type_map[uncertainty_type]))]
                mean = np.mean(add_scores)
                std = np.std(add_scores)
                bin_values, _, patches = plt.hist(add_scores, bins, alpha=0.5, density=density, label=add_set_id)
                plt.errorbar(mean, max(bin_values) * 1.01, xerr=std, color=patches[0].get_facecolor(), fmt="|")
            plt.legend()

        plt.ylabel("frequency")
        plt.xlabel("score")
        plt.title(f"\n{score_function_id}-score over {set_id} for {uncertainty_type}") # the \n is a quick and dirty fix for displying it
        plt.tight_layout()
        plt.savefig("score_histogram.png")

        # clean up before next plot
        plt.clf()
        plt.cla()

    def render_heatmap(self, params):
        pass



if __name__ == "__main__":

    server_address = ('', SERVER_PORT) # localhost:8000

    httpd = HTTPServer(server_address=server_address, RequestHandlerClass=CustomHandler)
    
    print(f"Server running at http://localhost:{SERVER_PORT}/")
    
    httpd.serve_forever()
