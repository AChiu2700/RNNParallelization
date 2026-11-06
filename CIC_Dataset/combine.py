import numpy as np
import pandas as pd

monday = pd.read_csv("Monday-WorkingHours.pcap_ISCX.csv")
tuesday = pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv")
wednesday = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")
thursday1 = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
thursday2 = pd.read_csv("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
friday1 = pd.read_csv("Friday-WorkingHours-Morning.pcap_ISCX.csv")
friday2 = pd.read_csv("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
friday3 = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

result = pd.concat([monday, tuesday, wednesday, thursday1, thursday2, friday1, friday2, friday3], ignore_index=True)

# index=False: row index would add a leading column correlated with label
# (files concatenated benign Monday -> attack-heavy Friday) and shift columns by one.
result.to_csv("combined.csv", index=False)
