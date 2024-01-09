class ModelPerformanceLog:
    def __init__(self):
        self.log = {}  # Dictionary to store data: {model_name: {dataset_name: {metric_name: [values]}}}

    def add_model(self, model_name):
        if model_name not in self.log:
            self.log[model_name] = {}

    def add_dataset(self, model_name, dataset_name):
        if model_name not in self.log:
            self.add_model(model_name)
        if dataset_name not in self.log[model_name]:
            self.log[model_name][dataset_name] = {}

    def add_metric(self, model_name, dataset_name, metric_name, values, trial=0):
        if model_name not in self.log:
            self.add_model(model_name)
        if dataset_name not in self.log[model_name]:
            self.add_dataset(model_name, dataset_name)
        if metric_name not in self.log[model_name][dataset_name]:
            self.log[model_name][dataset_name][metric_name] = {}

        if trial not in self.log[model_name][dataset_name][metric_name]:
            self.log[model_name][dataset_name][metric_name][trial] = []

        self.log[model_name][dataset_name][metric_name][trial].extend(values)

    def get_metric_values(self, model_name, dataset_name, metric_name, trial=0):
        if model_name in self.log and dataset_name in self.log[model_name] \
                and metric_name in self.log[model_name][dataset_name] \
                and trial in self.log[model_name][dataset_name][metric_name]:
            return self.log[model_name][dataset_name][metric_name][trial]
        else:
            return None

    def add_new_dataset(self, dataset_name):
        for model in self.log:
            self.add_dataset(model, dataset_name)

    def add_metric_for_dataset(self, model_name, dataset_name, metric_name, values, trial=0):
        if model_name in self.log and dataset_name in self.log[model_name]:
            if metric_name not in self.log[model_name][dataset_name]:
                self.log[model_name][dataset_name][metric_name] = {}
            if trial not in self.log[model_name][dataset_name][metric_name]:
                self.log[model_name][dataset_name][metric_name][trial] = []
            self.log[model_name][dataset_name][metric_name][trial].extend(values)


