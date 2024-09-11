import logging
import numpy as np
import pandas as pd
import time
from ananke.models.collection import Collection
from ananke.configurations.collection import HDF5StorageConfiguration

# Set logging level to INFO
logging.getLogger().setLevel(logging.INFO)

class TriggerDatasetProcessor:
    def __init__(self, path: str, ignore_records: bool = False):
        """Initialize the TriggerDatasetProcessor with the path to data and configuration."""
        self.path = path
        self.ignore_records = ignore_records
        self.config = HDF5StorageConfiguration(data_path=path, read_only=False)
        self.collection = Collection(self.config)
        self.hits = None
        self.times = None
        self.records = None
        self.trigger_threshhold=7
        self.trigger_intervals=100

    @staticmethod
    def count_in_interval(lst, lower_bound, upper_bound):
        """Counts the number of elements in a list that fall within a specified interval."""
        return sum(lower_bound <= x <= upper_bound for x in lst)

    @staticmethod
    def custom_agg(x):
        """Custom aggregation function to either sum lists or sum numerical values."""
        if isinstance(x.iloc[0], list):
            return sum(x, [])
        else:
            return x.sum()

    def load_data(self):
        """Load hit and source data from the collection."""
        self.collection.open()
        self.hits = self.collection.storage.get_hits()
        self.collection.close()

        upperlimit = self.hits.get_statistics().max
        lowerlimit = self.hits.get_statistics().min
        self.times = np.arange(lowerlimit, upperlimit + self.trigger_intervals, self.trigger_intervals)

        if self.ignore_records:
            self.records = [1]
        else:
            self.records = self.hits.df['record_id'].drop_duplicates()

    def prepare_trigger_data(self, hit):
        """
        Processes hit data to determine the number of hits in specified time intervals
        and filters based on conditions.
        """
        for i, j in zip(self.times, self.times[1:]):
            hit[f'hit_in_interval_{i}-{j}'] = hit['time'].apply(lambda x: self.count_in_interval(x, i, j))

        column_names = [col for col in hit.columns if col.startswith("hit_in_interval_")]

        trigger_data_list = []

        for name in column_names:
            trigger_data = hit[['string_id', 'module_id', 'pmt_id', 'time', name]]
            trigger_data = trigger_data[trigger_data[name] != 0]
            trigger_data = trigger_data.groupby(['string_id', 'module_id']).agg({'time': lambda x: sum(x, []), name: list}).reset_index()
            trigger_data[f'pmts_hit_in_interval_{name[18:]}'] = trigger_data[name].apply(len)
            trigger_data = trigger_data[trigger_data[f'pmts_hit_in_interval_{name[18:]}'] > self.trigger_threshhold]
            trigger_data[name] = trigger_data[name].apply(sum)
            trigger_data.rename(columns={name: f'ModuleHitCount{name[18:]}'}, inplace=True)
            trigger_data_list.append(trigger_data)

        return pd.concat(trigger_data_list, ignore_index=True, axis=0).fillna(0)

    def process_records(self):
        """
        Processes each record to compute the number of modules triggered and hits per interval.
        """
        NumberOfModsTriggeredPerRec = []
        HitsPerIntervalPerRec = []
        HitTiming = []

        for record in self.records:
            print(f'record: {record} started')
            time_per_record = time.time()

            hit = self.hits if self.ignore_records else self.hits.get_by_record_ids(record)
            hit = hit.df.groupby(["string_id", 'module_id', 'pmt_id'])['time'].apply(list).reset_index()

            StackedTriggerData = self.prepare_trigger_data(hit)

            if StackedTriggerData.shape[0] > 1:
                StackedTriggerData = StackedTriggerData.groupby(['string_id', 'module_id']).agg(self.custom_agg).reset_index()

            hits_per_interval = []
            number_of_modules_triggered = []

            for col in StackedTriggerData.columns:
                if col.startswith("ModuleHitCount"):
                    number_of_modules_triggered.append(StackedTriggerData[StackedTriggerData[col] > 0].shape[0])
                    hits_per_interval.append(sum(StackedTriggerData[col]))

            NumberOfModsTriggeredPerRec.append(number_of_modules_triggered)
            HitsPerIntervalPerRec.append(hits_per_interval)
            HitTiming.append(StackedTriggerData['time'])

        return NumberOfModsTriggeredPerRec, HitsPerIntervalPerRec, HitTiming

    def run(self):
        """Main function to compute trigger dataset statistics."""
        start_time = time.time()
        self.load_data()
        NumberOfModsTriggeredPerRec, HitsPerIntervalPerRec, HitTiming = self.process_records()
        print(f'Total time taken: {time.time() - start_time}')
        return NumberOfModsTriggeredPerRec, HitsPerIntervalPerRec, HitTiming


# Example usage
if __name__ == "__main__":
    processor = TriggerDatasetProcessor(path="your_data_path.h5", ignore_records=False)
    NumberOfModsTriggeredPerRec, HitsPerIntervalPerRec, HitTiming = processor.run()
