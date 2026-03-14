CREATE OR REPLACE PROCEDURE pc_etl_drug_exposure(input_start_date TEXT, input_end_date TEXT)
 LANGUAGE plpgsql
AS $procedure$
	DECLARE
		-- 변환 기간 정의
		start_date timestamp;
		end_date timestamp;
	BEGIN
		start_date := TO_DATE(input_start_date, 'YYYYMMDD');
		end_date := TO_DATE(input_end_date, 'YYYYMMDD');
		
		-- 작업 스키마 설정
		SET SCHEMA '';
	
		-- drug_exposure_id관리를 위한 seq 생성
		CREATE SEQUENCE IF NOT EXISTS seq_drug_exposure_id;


        INSERT INTO cdm.drug_exposure (
            person_id, drug_concept_id, drug_exposure_start_date,
            drug_source_value, drug_source_concept_id
        )
        SELECT
            pe.person_id,
            COALESCE(b.concept_id, 0)       AS drug_concept_id,
            s.order_date                     AS drug_exposure_start_date,
            s.drug_code                      AS drug_source_value,
            COALESCE(b.source_concept_id, 0)       AS drug_source_concept_id
        FROM source.prescription s
        INNER JOIN
            person pe
            ON s.patient_id = pe.person_source_value
        INNER JOIN
            local_edi b
            ON s.ordercode = b.ordercode
            AND b.final_domain = 'Drug'
            AND orderdate between fromdate and todate
        LEFT JOIN
            provider pr
            ON s.dr = pr.provider_source_value
        LEFT JOIN
            visit_occurrence v
            ON pe.patient_id = v.person_id
            AND s.medtime = visit_start_datetime
            AND s.visitfg = v.visit_source_value
            AND s.meddept = v.meddept
        WHERE s.order_date >= :start_date;
    COMMIT ;
	END 
	$procedure$
;

CALL pc_etl_drug_exposure('20100101', '20221231');
