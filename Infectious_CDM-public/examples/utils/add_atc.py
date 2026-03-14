import pandas as pd
from DataTransformer import DataTransformer 
import os

config = DataTransformer("config.yaml").load_config("config.yaml")

atccode = [
    ['A02BC51', 'omeprazole, combinations'],
    ['A02BX16', 'irsogladine'],
    ['A07FA03', 'escherichia coli'],
    ['A10BD28', 'metformin and teneligliptin'],
    ['A10BD29', 'sitagliptin and dapagliflozin'],
    ['B01AC28', 'limaprost'],
    ['C04AX33', 'clazosentan'],
    ['C10BX20', 'rosuvastatin and telmisartan'],
    ['G04BD15', 'vibegron'],
    ['H05BX06', 'evocalcet'],
    ['L01FX24', 'teclistamab'],
    ['L01FX25', 'mosunetuzumab'],
    ['L01FX28', 'glofitamab'],
    ['L01FY01', 'pertuzumab and trastuzumab'],
    ['L01XK52', 'niraparib and abiraterone'],
    ['L02BX53', 'abiraterone and prednisolone'],
    ['L03AA19', 'eflapegrastim'],
    ['L04AC24', 'mirikizumab'],
    ['L04AE01', 'fingolimod'],
    ['L04AE02', 'ozanimod'],
    ['L04AE04', 'ponesimod'],
    ['L04AF01', 'tofacitinib'],
    ['L04AF02', 'baricitinib'],
    ['L04AF03', 'upadacitinib'],
    ['L04AF04', 'filgotinib'],
    ['L04AF06', 'peficitinib'],
    ['L04AF07', 'deucravacitinib'],
    ['L04AG03', 'natalizumab'],
    ['L04AG04', 'belimumab'],
    ['L04AG05', 'vedolizumab'],
    ['L04AG06', 'alemtuzumab'],
    ['L04AG10', 'inebilizumab'],
    ['L04AH01', 'sirolimus'],
    ['L04AH02', 'everolimus'],
    ['L04AJ01', 'eculizumab'],
    ['L04AJ02', 'ravulizumab'],
    ['L04AJ04', 'sutimlimab'],
    ['L04AJ05', 'avacopan'],
    ['L04AK01', 'leflunomide'],
    ['L04AK02', 'teriflunomide'],
    ['N02AJ22', 'hydrocodone and paracetamol'],
    ['N02CC51', 'sumatriptan and naproxen']
]

atc_df = pd.DataFrame(atccode, columns = ["ATC 코드", "ATC 코드명"])
df = pd.read_csv(os.path.join(config["CDM_path"], "atcindex.csv"))
df = df[["ATC 코드", "ATC 코드명"]]
# df.to_csv(os.path.join(config["CDM_path"], "atcindex.csv"), index=False)

result = pd.concat([df, atc_df], axis=0)
result.to_csv(os.path.join(config["CDM_path"], "atcindex.csv"), index=False)