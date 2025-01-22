

class Logger:
    # NOTE: do not move this file into other folder for refactoring for example utils.py or etc.
    def __init__(self, log_file, dataset_yaml):

        self.log_file = log_file
        self.log_script_file = log_file.replace("log.txt", "log_script.py")
        self.log_data_file = log_file.replace("log.txt", "log_data.txt")

        with open(log_file, "w") as f:
            pass
        with open(self.log_script_file, "w") as f:
            pass
        with open(self.log_data_file, "w") as f:
            pass

        self.dataset_yaml = dataset_yaml

        self.log_script()
        self.log_data()

        self.step = 0

    def log(self, message):
        self.step += 1
        with open(self.log_file, "a") as f:
            f.write(f"{self.step} " + message)

    def log_script(self):
        # Open current file and log every lines of code inside the file
        with open(__file__, "r") as f:
            lines = f.readlines()

        with open(self.log_script_file, "a") as f2:
            f2.writelines(lines)

    def log_data(self):
        with open(self.dataset_yaml, "r") as f:
            lines = f.readlines()

        with open(self.log_data_file, "a") as f:
            f.writelines(lines)

    def resume(self, resume_file):  # NOT USED
        # Open the existing log file to read its content
        with open("/".join(resume_file.split("/")[:-1]) + "/log.txt", "r") as f:
            lines = f.readlines()  # Read all lines

        # Write the content to the new log file
        with open(self.log_file, "w") as f2:
            f2.writelines(lines)

        # Get the last step number from the lines read
        if lines:
            self.step = int(lines[-1].split(" ")[0])