from src.aggregation import build_density_table, save_density

def main():
    density = build_density_table("data/raw/CityC_Sapporo.csv")
    print(density.head(10))
    save_density(density, "data/processed/sapporo_density.parquet")
    print("Saved -> data/processed/sapporo_density.parquet")

if __name__ == "__main__":
    main()
