import pandas as pd
import numpy as np

#should be rewritten to use less memory
# storing each data set doesn't need to happen a loop would be more effective
def get_ddos_data_cleaned(path, num_rows):
    # collect data from each file currently only using data from 1/12
    DNS = pd.read_csv(path + 'DrDoS_DNS.csv', low_memory=False, nrows=num_rows)
    LDAP = pd.read_csv(path + 'DrDoS_LDAP.csv', low_memory=False, nrows=num_rows)
    MSSQL = pd.read_csv(path + 'DrDoS_MSSQL.csv', low_memory=False, nrows=num_rows)
    NTP = pd.read_csv(path + 'DrDoS_NTP.csv', low_memory=False, nrows=num_rows)
    NETBIOS = pd.read_csv(path + 'DrDoS_NetBIOS.csv', low_memory=False, nrows=num_rows)
    SNMP = pd.read_csv(path + 'DrDoS_SNMP.csv', low_memory=False, nrows=num_rows)
    SSDP = pd.read_csv(path + 'DrDoS_SSDP.csv', low_memory=False, nrows=num_rows)
    UDP = pd.read_csv(path + 'DrDoS_UDP.csv', low_memory=False, nrows=num_rows)
    SYN = pd.read_csv(path + 'Syn.csv', low_memory=False, nrows=num_rows)
    TFTP = pd.read_csv(path + 'TFTP.csv', low_memory=False, nrows=num_rows)
    UDPLag= pd.read_csv(path + 'UDPLag.csv', low_memory=False, nrows=num_rows)
    dataset1 = pd.concat([DNS, LDAP,MSSQL,NTP, NETBIOS, SNMP, SSDP, UDP, SYN, TFTP, UDPLag])
    
    # Rename the types of attacks(aka labels)
    dataset1.rename(columns={' Label':'Label'},inplace=True) # Remove space in label column name
    old_labels = ['DrDoS_SSDP', 'DrDoS_LDAP', 'DrDoS_SNMP', 'DrDoS_NetBIOS', 'DrDoS_MSSQL', 'DrDoS_UDP', 'DrDoS_DNS', 'DrDoS_NTP']
    new_labels = ['SSDP', 'LDAP', 'SNMP', 'NetBIOS', 'MSSQL', 'UDP', 'DNS', 'NTP']
    dataset1['Label'].replace(old_labels, new_labels, inplace=True)


    # Remove overfitting columns (the same ones used in the original paper)
    overfitting_columns = ['Unnamed: 0', ' Source IP', ' Destination IP', ' Source Port', ' Destination Port', ' Timestamp', 'SimillarHTTP', 'Flow ID']
    dataset1.drop(labels=overfitting_columns, axis='columns', inplace=True)

    # Drop NaN values
    dataset1.dropna(axis='index', inplace=True)
    dataset1 = dataset1[~dataset1.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    # Remove columns with only values of 0
    useless_columns = [' Bwd PSH Flags', ' Fwd URG Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Bwd URG Flags', 'FIN Flag Count',
                    ' PSH Flag Count', ' ECE Flag Count', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate',
                    ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
    dataset1.drop(labels=useless_columns, axis='columns', inplace=True)

    return dataset1