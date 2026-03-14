SELECT 'drug_exposure' AS table_name,
       (SELECT COUNT(*) FROM source.prescription
        WHERE order_date >= start_date)    AS source_count,
       (SELECT COUNT(*) FROM cdm.drug_exposure
        WHERE drug_exposure_start_date >= start_date) AS cdm_count;