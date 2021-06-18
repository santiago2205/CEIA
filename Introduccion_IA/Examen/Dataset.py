import pickle
import csv
import numpy as np


class Dataset:
    instance = None
    data = None

    def __new__(cls, filename):
        if Dataset.instance is None:
            print("Creating new Dataset instance")
            Dataset.instance = super(Dataset, cls).__new__(cls)
            return Dataset.instance
        else:
            return Dataset.instance

    def __init__(self, filename):
        print("Initialising Dataset")

        try:
            with open(filename + '.pkl', 'rb') as pkl_file:
                self.data = pickle.load(pkl_file)
        except FileNotFoundError:
            print("CSV file found. Building PKL file...")
            try:
                with open(filename + '.csv') as csv_file:
                    with open(filename + '.pkl', 'wb') as pkl_file:

                        csv_reader = csv.reader(csv_file, delimiter=',')

                        def generator(reader):
                            first_skipped = False
                            for line in reader:
                                if not first_skipped:
                                    first_skipped = True
                                    continue
                                yield line[0], line[1]

                        gen = generator(csv_reader)

                        structure = [('entrada', np.float32),
                                     ('salida', np.float32)]

                        array = np.fromiter(gen, dtype=structure)

                        pickle.dump(array, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

                    pkl_file.close()

                with open(filename + '.pkl', 'rb') as pkl_file:
                    self.data = pickle.load(pkl_file)
            except FileNotFoundError:
                print("No PKL or CSV named " + filename + " was found.")
            finally:
                csv_file.close()
        finally:
            pkl_file.close()

