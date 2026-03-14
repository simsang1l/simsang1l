from sqlalchemy import Column, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import UserDefinedType

# 커스텀 Vector 타입 정의
class VectorType(UserDefinedType):
    def __init__(self, dimension):
        self.dimension = dimension

    def get_col_spec(self):
        return f"vector({self.dimension})"

    def bind_expression(self, bindvalue):
        return bindvalue

    def column_expression(self, col):
        return col

    def bind_processor(self, dialect):
        def process(value):
            if isinstance(value, list):
                return f'[{", ".join(map(str, value))}]'
            return value
        return process
    
# 기본 클래스 생성
Base = declarative_base()

def create_dicom_key_class(table_name):
    class DicomKey(Base):
        __tablename__ = table_name
        __table_args__ = {"schema": "your_schema"}

        id = Column(Integer, primary_key=True, autoincrement=True)
        patient_no = Column(Text, nullable=True)
        workdate = Column(Text, nullable=False)
        execdate = Column(Text, nullable=True)
        exectime = Column(Text, nullable=True)
        file_size =  Column(Text, nullable=True)
        source_filepath = Column(Text, nullable=False)
        source_filename = Column(Text, nullable=True)
        source_file_extension =  Column(Text, nullable=True)
        source_modality = Column(Text, nullable=True)
        accessionnumber = Column(Text, nullable=True) # Accession_Number
        studyinstanceuid = Column(Text, nullable=True) # Study_Instance_UID
        studydate = Column(Text, nullable=True) # StudyDate
        studytime = Column(Text, nullable=True) # StudyTime
        studydescription = Column(Text, nullable=True) # StudyDescription
        manufacturer = Column(Text, nullable=True) # Manufacturer
        modality = Column(Text, nullable=True) # Modality
        seriesdate = Column(Text, nullable=True) # SeriesDate
        seriestime = Column(Text, nullable=True) # SeriesTime
        seriesinstanceuid = Column(Text, nullable=True) # Series_Instance_UID
        bodypartexamined = Column(Text, nullable=True) # BodyPartExamined
        laterality = Column(Text, nullable=True) # Laterality
        patientposition = Column(Text, nullable=True) # PatientPosition
        slicethickness = Column(Text, nullable=True) # SliceThickness
        rows = Column(Text, nullable=True) # Rows
        columns = Column(Text, nullable=True) # Columns
        windowcenter = Column(Text, nullable=True) # WindowCenter
        windowwidth = Column(Text, nullable=True) # WindowWidth
        seriesdescription = Column(Text, nullable=True) # SeriesDescription
        kvp = Column(Text, nullable=True) # kVP
        requestedproceduredescription = Column(Text, nullable = True)# RequestedProcedureDescription
        viewposition = Column(Text, nullable=True) # ViewPosition
        protocolname = Column(Text, nullable=True) # ProtocolName
        scanningsequence = Column(Text, nullable=True) # ScanningSequence
        repetitiontime = Column(Text, nullable=True) # RepetitionTime
        acquisitioncontrast = Column(Text, nullable=True) # AcquisitionContrast
        seriesnumber = Column(Text, nullable=True) # SeriesNumber
        instancenumber = Column(Text, nullable=True) # InstanceNumber
        bodypartexamined = Column(Text, nullable=True) # BodyPartExamined
        studydescription_embedding = Column(VectorType(384), nullable=True) # StudyDescription Embedding
        requestedproceduredescription_embedding = Column(VectorType(384), nullable = True)# RequestedProcedureDescription Embedding
        rename_filepath = Column(Text, nullable=True)
        person_id = Column(Integer, nullable=True)
        orddate = Column(Text, nullable=True)
        ordseqno = Column(Text, nullable=True)
        ordname = Column(Text, nullable=True)
        label = Column(Text, nullable=True)

    return DicomKey