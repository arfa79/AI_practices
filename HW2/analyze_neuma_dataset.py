import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind

class NeuMaDatasetAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.subjects = {}
        self.aggregate_results = []

    def load_subject_data(self, subject_id):
        mat_file = os.path.join("D:\\arfa\\code\\Hw1\\hw2", 'S01.mat')
        try:
            # Load the MAT file and access ET_clean data directly
            mat_data = scipy.io.loadmat(mat_file)
            eye_data = mat_data['S01'][0,0]['ET_clean'][0,0][0]  # This gives us the (6, 43200) array
            
            print(f"\nProcessing subject {subject_id}")
            print(f"Data array shape: {eye_data.shape}")
            
            # Count valid (non-NaN) samples
            valid_samples = np.sum(~np.isnan(eye_data[0]))
            print(f"Number of valid samples: {valid_samples}")
            
            # Create the gaze data dictionary
            gaze_data = {
                'left_x': eye_data[0],  # First row
                'left_y': eye_data[1],  # Second row
                'left_pupil': eye_data[2],  # Third row
                'right_x': eye_data[3],  # Fourth row
                'right_y': eye_data[4],  # Fifth row
                'right_pupil': eye_data[5]  # Sixth row
            }
            
            self.subjects[subject_id] = gaze_data
            return True
                
        except Exception as e:
            print(f"Error loading subject {subject_id}: {str(e)}")
            return False

    def plot_gaze_positions(self, subject_id):
        if subject_id not in self.subjects:
            print(f"Subject {subject_id} data not loaded")
            return
        
        data = self.subjects[subject_id]
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot left eye gaze positions
        ax1.scatter(data['left_x'], data['left_y'], alpha=0.5, s=1)
        ax1.set_title('Left Eye Gaze Positions')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        
        # Plot right eye gaze positions
        ax2.scatter(data['right_x'], data['right_y'], alpha=0.5, s=1, color='red')
        ax2.set_title('Right Eye Gaze Positions')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        
        plt.tight_layout()
        plt.savefig(f'subject_{subject_id}_gaze_positions.png')
        plt.close()

    def plot_pupil_sizes(self, subject_id):
        if subject_id not in self.subjects:
            print(f"Subject {subject_id} data not loaded")
            return
        
        data = self.subjects[subject_id]
        time_points = np.arange(len(data['left_pupil']))
        
        plt.figure(figsize=(15, 6))
        plt.plot(time_points, data['left_pupil'], label='Left Eye', alpha=0.7)
        plt.plot(time_points, data['right_pupil'], label='Right Eye', alpha=0.7)
        plt.title(f'Pupil Size Variations Over Time - Subject {subject_id}')
        plt.xlabel('Time Points')
        plt.ylabel('Pupil Size')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'subject_{subject_id}_pupil_sizes.png')
        plt.close()

    def analyze_fixations(self, subject_id):
        if subject_id not in self.subjects:
            print(f"Subject {subject_id} data not loaded")
            return

        # Simulated fixation data for demonstration
        fixation_data = {
            'Product_ID': [f'Product{i}' for i in range(1, 11)],
            'Bought_Status': np.random.choice(['Bought', 'NotBought'], 10),
            'Fixation_Count': np.random.randint(5, 15, 10),
            'Total_Fixation_Duration': np.random.randint(500, 2000, 10)
        }
        df = pd.DataFrame(fixation_data)
        df.to_csv(f'fixation_analysis_subject_{subject_id}.csv', index=False)

        # Boxplot
        plt.figure(figsize=(10, 6))
        df.boxplot(column=['Fixation_Count', 'Total_Fixation_Duration'], by='Bought_Status', grid=False)
        plt.suptitle(f'Fixation Analysis - Subject {subject_id}')
        plt.savefig(f'plots/subject_{subject_id}_fixation_analysis.png')
        plt.close()

    def analyze_pupil_sizes(self, subject_id):
        if subject_id not in self.subjects:
            print(f"Subject {subject_id} data not loaded")
            return

        # Simulated pupil size data for demonstration
        pupil_data = {
            'Pupil_Size': np.random.normal(400, 20, 100),
            'Bought_Status': np.random.choice(['Bought', 'NotBought'], 100)
        }
        df = pd.DataFrame(pupil_data)

        # Violin plot
        plt.figure(figsize=(10, 6))
        df.boxplot(column=['Pupil_Size'], by='Bought_Status', grid=False)
        plt.suptitle(f'Pupil Size Analysis - Subject {subject_id}')
        plt.savefig(f'plots/subject_{subject_id}_pupil_size_analysis.png')
        plt.close()

        # Statistical summary
        bought = df[df['Bought_Status'] == 'Bought']['Pupil_Size']
        not_bought = df[df['Bought_Status'] == 'NotBought']['Pupil_Size']
        t_stat, p_value = ttest_ind(bought, not_bought)
        summary = (
            f"Bought Products: Mean pupil size = {bought.mean():.2f}px (±{bought.std():.2f})\n"
            f"Not Bought: Mean pupil size = {not_bought.mean():.2f}px (±{not_bought.std():.2f})\n"
            f"t-test p-value = {p_value:.4f}"
        )
        with open(f'plots/subject_{subject_id}_pupil_size_summary.txt', 'w') as f:
            f.write(summary)

    def analyze_temporal_alignment(self, subject_id):
        if subject_id not in self.subjects:
            print(f"Subject {subject_id} data not loaded")
            return

        # Simulated temporal alignment data for demonstration
        time_points = np.arange(0, 1000, 10)
        pupil_sizes = np.random.normal(400, 20, len(time_points))
        bought_markers = np.random.choice(time_points, 10, replace=False)

        plt.figure(figsize=(15, 6))
        plt.plot(time_points, pupil_sizes, label='Pupil Size')
        for marker in bought_markers:
            plt.axvline(marker, color='blue', linestyle='--', alpha=0.7, label='Bought Fixation')
        plt.title(f'Temporal Alignment - Subject {subject_id}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Pupil Size')
        plt.legend()
        plt.savefig(f'plots/subject_{subject_id}_temporal_alignment.png')
        plt.close()

    def analyze_all_subjects(self):
        for subject_id in range(1, 5):
            if self.load_subject_data(subject_id):
                self.analyze_fixations(subject_id)
                self.analyze_pupil_sizes(subject_id)
                self.analyze_temporal_alignment(subject_id)

if __name__ == "__main__":
    print("Starting NeuMa Dataset Analysis...")
    analyzer = NeuMaDatasetAnalyzer(".")
    analyzer.analyze_all_subjects()
    print("\nAnalysis complete.")