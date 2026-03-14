from DataTransformer import *
import logging

if __name__ == "__main__":
    config = "config.yaml"

    start_time = datetime.now()
    try : 
        care_site = CareSiteTransformer(config)    
        care_site.transform()

        provider = ProviderTransformer(config)    
        provider.transform()

        person = PersonTransformer(config)    
        person.transform()

        visit_occurrence = VisitOccurrenceTransformer(config)    
        visit_occurrence.transform()

        visit_detail = VisitDetailTransformer(config)    
        visit_detail.transform()
        
        local_kcd = LocalKCDTransformer(config)    
        local_kcd.transform()

        condition_occurrence = ConditionOccurrenceTransformer(config)    
        condition_occurrence.transform()

        local_edi = LocalEDITransformer(config)    
        local_edi.transform()

        ## drug_exposure = DrugexposureTransformer(config)    
        ## drug_exposure.transform()

        drug_exposure = DrugexposureChunkTransformer(config)    
        drug_exposure.transform()

        ## measurement_stresult = MeasurementStexmrstTransformer(config)
        ## measurement_stresult.transform()

        measurement_stresult = MeasurementStexmrstChunkTransformer(config)
        measurement_stresult.transform()

        measurement_bmi = MeasurementVSTransformer(config)
        measurement_bmi.transform()

        merge_measurement = MergeMeasurementTransformer(config)
        merge_measurement.transform()                      

        procedure_trt = ProcedureOrderTransformer(config)
        procedure_trt.transform()

        procedure_stresult = ProcedureStexmrstTransformer(config)
        procedure_stresult.transform()
        
        merge_procedure = MergeProcedureTransformer(config)
        merge_procedure.transform()

        observation_period = ObservationPeriodTransformer(config)
        observation_period.transform()
        
        
    except Exception as e :
        logging.error(f"Exucution failed: {e}", exc_info=True)
    
    logging.info(f"transform and Load end, elapsed_time is : {datetime.now() - start_time}")