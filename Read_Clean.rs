use serde::Deserialize;
use ndarray::{Array1, Array2, array, s};
use rand::Rng;

// This module has the functions for reading and cleaning the dataset 
// before it is used for the regression model.

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
// The DirtyRecord struct takes a dataset and has every value be a string as prep
//for cleaning the dataset
pub struct DirtyRecord {
    Id: String,
    ProductionBudget: String,
    DomesticGross: String,
    WorldwideGross: String,
    DomesticBoxOffice: String,
    InternationalBoxOffice: String,
    WorldwideBoxOffice: String,
    DomesticDVDSales: String,
    DomesticBluRaySales: String,
    TotalDomesticVideoSales: String,
    OpeningWeekend: String,
    Legs: String,
    InflAdjDomBO: String,
    RunningTime: String,
}

#[derive(Debug, Default)]
// CleanRecord is the desired types for every column in the dataset / clean version of the dataset
pub struct CleanRecord {
    pub Id: u64,
    pub ProductionBudget: u128,
    pub DomesticGross: u128,
    pub WorldwideGross: u128,
    pub DomesticBoxOffice: u128,
    pub InternationalBoxOffice: u128,
    pub WorldwideBoxOffice: u128,
    pub DomesticDVDSales: u128,
    pub DomesticBluRaySales: u128,
    pub TotalDomesticVideoSales: u128,
    pub OpeningWeekend: u128,
    pub Legs: f64,
    pub InflAdjDomBO: u128,
    pub RunningTime: u64,
}

// Converts DirtyRrcord to CleanRecord and handles missing values
pub fn clean_record(r: DirtyRecord) -> CleanRecord {
    let mut c = CleanRecord::default();
    c.Id = r.Id.replace(',', "").parse::<u64>().unwrap();
    c.ProductionBudget = r.ProductionBudget.parse::<u128>().unwrap();
    c.DomesticGross = r.DomesticGross.parse::<u128>().unwrap();
    c.WorldwideGross = r.WorldwideGross.parse::<u128>().unwrap();
    let DomesticBoxOffice = r.DomesticBoxOffice.parse::<u128>();
    if DomesticBoxOffice.is_ok() {
        c.DomesticBoxOffice = DomesticBoxOffice.unwrap();
    } else {
        c.DomesticBoxOffice = 0;
    }
    let InternationalBoxOffice = r.InternationalBoxOffice.parse::<u128>();
    if InternationalBoxOffice.is_ok() {
        c.InternationalBoxOffice = InternationalBoxOffice.unwrap();
    } else {
        c.InternationalBoxOffice = 0;
    }
    let WorldwideBoxOffice = r.WorldwideBoxOffice.parse::<u128>();
    if WorldwideBoxOffice.is_ok() {
        c.WorldwideBoxOffice = WorldwideBoxOffice.unwrap();
    } else {
        c.WorldwideBoxOffice = 0;
    }
    let DomesticDVDSales = r.DomesticDVDSales.parse::<u128>();
    if DomesticDVDSales.is_ok() {
        c.DomesticDVDSales = DomesticDVDSales.unwrap();
    } else {
        c.DomesticDVDSales = 0;
    }
    let DomesticBluRaySales = r.DomesticBluRaySales.parse::<u128>();
    if DomesticBluRaySales.is_ok() {
        c.DomesticBluRaySales = DomesticBluRaySales.unwrap();
    } else {
        c.DomesticBluRaySales = 0;
    }
    let TotalDomesticVideoSales = r.TotalDomesticVideoSales.parse::<u128>();
    if TotalDomesticVideoSales.is_ok() {
        c.TotalDomesticVideoSales = TotalDomesticVideoSales.unwrap();
    } else {
        c.TotalDomesticVideoSales = 0;
    }
    let OpeningWeekend = r.OpeningWeekend.parse::<u128>();
    if OpeningWeekend.is_ok() {
        c.OpeningWeekend = OpeningWeekend.unwrap();
    } else {
        c.OpeningWeekend = 0;
    }
    let Legs = r.Legs.parse::<f64>();
    if Legs.is_ok() {
        c.Legs = Legs.unwrap();
    } else {
        c.Legs = 0.0;
    }
    let InflAdjDomBO = r.InflAdjDomBO.parse::<u128>();
    if InflAdjDomBO.is_ok() {
        c.InflAdjDomBO = InflAdjDomBO.unwrap();
    } else {
        c.InflAdjDomBO = 0;
    }
    let RunningTime = r.RunningTime.parse::<u64>();
    if RunningTime.is_ok() {
        c.RunningTime = RunningTime.unwrap();
    } else {
        c.RunningTime = 0;
    }
    return c;
}

// Filter out invalid CleanRecords where WorldwideGross == 0
pub fn filter_clean_records(records: Vec<CleanRecord>) -> Vec<CleanRecord> {
    records.into_iter()
           .filter(|record| record.WorldwideGross != 0)
           .collect()
}

// Read, clean, and filter record into a vector
pub fn process_csv_file() -> Vec<CleanRecord> {
    let mut rdr = csv::Reader::from_path("src/TopMovies.csv").unwrap();
    let mut cleanv: Vec<CleanRecord> = Vec::new();

    for result in rdr.deserialize::<DirtyRecord>() {
        match result {
            Ok(record) => {
                let cleanrec = clean_record(record);
                cleanv.push(cleanrec);
            }
            Err(err) => {
                println!("Error parsing record: {}", err);
            }
        }
    }
    
    filter_clean_records(cleanv)
}

// Converts cleaned record to ndarray format with added noise
pub fn convert_to_xy(records: Vec<CleanRecord>, noise_stddev: f64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng();
    let mut all_data: Vec<f64> = Vec::new();

    for record in &records {
        all_data.push(record.ProductionBudget as f64);
        all_data.push(record.DomesticGross as f64);
        all_data.push(record.DomesticBoxOffice as f64);
        all_data.push(record.InternationalBoxOffice as f64);
        all_data.push(record.DomesticDVDSales as f64);
        all_data.push(record.DomesticBluRaySales as f64);
        all_data.push(record.TotalDomesticVideoSales as f64);
        all_data.push(record.OpeningWeekend as f64);
        all_data.push(record.Legs);
        all_data.push(record.InflAdjDomBO as f64);
        all_data.push(record.RunningTime as f64);

        // Add noise to the target variable (WorldwideGross) to avoid overfitting
        let noisy_worldwide_gross = record.WorldwideGross as f64 + rng.gen_range(-noise_stddev..noise_stddev);

        
        all_data.push(noisy_worldwide_gross);
    }

    let num_rows = records.len();
    let num_cols = 12; 
    
    let all_features = Array2::<f64>::from_shape_vec((num_rows, num_cols), all_data).unwrap();

    let y = all_features.column(11).to_owned(); 

    let x = all_features.slice(s![.., 0..10]).to_owned(); 

    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_record_basic() {
        let dirty = DirtyRecord {
            Id: "1".to_string(),
            ProductionBudget: "100000000".to_string(),
            DomesticGross: "50000000".to_string(),
            WorldwideGross: "150000000".to_string(),
            DomesticBoxOffice: "50000000".to_string(),
            InternationalBoxOffice: "100000000".to_string(),
            WorldwideBoxOffice: "150000000".to_string(),
            DomesticDVDSales: "2000000".to_string(),
            DomesticBluRaySales: "3000000".to_string(),
            TotalDomesticVideoSales: "5000000".to_string(),
            OpeningWeekend: "25000000".to_string(),
            Legs: "2.5".to_string(),
            InflAdjDomBO: "55000000".to_string(),
            RunningTime: "120".to_string(),
        };

        let clean = clean_record(dirty);
        assert_eq!(clean.ProductionBudget, 100_000_000);
        assert_eq!(clean.WorldwideGross, 150_000_000);
        assert_eq!(clean.Legs, 2.5);
        assert_eq!(clean.RunningTime, 120);
    }
}