#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <iomanip>

using namespace std;

// Data Point ka Structure
struct DataPoint {
    vector<double> features; // Sepal Length, Sepal Width, Petal Length, Petal Width
    string label;            // Flower Species (Setosa, Versicolor, Virginica)
};

// Euclidean Distance Calculate Karne Ka Function
double calculateDistance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
}

// KNN Classification Function
string classifyKNN(const vector<DataPoint>& trainData, const vector<double>& testInstance, int k) {
    // Distance aur Index store karne ke liye
    vector<pair<double, string>> distances;

    for (const auto& point : trainData) {
        double dist = calculateDistance(point.features, testInstance);
        distances.push_back({dist, point.label});
    }

    // Shortest distance ke hisab se sort karna
    sort(distances.begin(), distances.end());

    // Top 'K' nearest neighbors ke votes count karna
    map<string, int> classVotes;
    for (int i = 0; i < k; ++i) {
        classVotes[distances[i].second]++;
    }

    // Highest vote wali class dhoondna
    string bestClass = "";
    int maxVotes = -1;
    for (const auto& vote : classVotes) {
        if (vote.second > maxVotes) {
            maxVotes = vote.second;
            bestClass = vote.first;
        }
    }

    return bestClass;
}

int main() {
    cout << "========================================================\n";
    cout << "   DecodeLabs Project 2: AI Data Classification (C++)   \n";
    cout << "========================================================\n\n";

    // Sample Iris Dataset (Training Data)
    vector<DataPoint> trainData = {
        // Setosa Samples
        {{5.1, 3.5, 1.4, 0.2}, "Setosa"},
        {{4.9, 3.0, 1.4, 0.2}, "Setosa"},
        {{4.7, 3.2, 1.3, 0.2}, "Setosa"},
        {{4.6, 3.1, 1.5, 0.2}, "Setosa"},
        
        // Versicolor Samples
        {{7.0, 3.2, 4.7, 1.4}, "Versicolor"},
        {{6.4, 3.2, 4.5, 1.5}, "Versicolor"},
        {{6.9, 3.1, 4.9, 1.5}, "Versicolor"},
        {{5.5, 2.3, 4.0, 1.3}, "Versicolor"},

        // Virginica Samples
        {{6.3, 3.3, 6.0, 2.5}, "Virginica"},
        {{5.8, 2.7, 5.1, 1.9}, "Virginica"},
        {{7.1, 3.0, 5.9, 2.1}, "Virginica"},
        {{6.3, 2.9, 5.6, 1.8}, "Virginica"}
    };

    // Test Dataset (Validation Ke Liye)
    vector<DataPoint> testData = {
        {{5.0, 3.4, 1.5, 0.2}, "Setosa"},      // Actual Setosa
        {{6.0, 2.9, 4.5, 1.5}, "Versicolor"},  // Actual Versicolor
        {{6.5, 3.0, 5.2, 2.0}, "Virginica"}    // Actual Virginica
    };

    int K = 3; // K-Nearest Neighbors
    int correctPredictions = 0;

    cout << "Testing K-Nearest Neighbors Model (K = " << K << "):\n";
    cout << "--------------------------------------------------------\n";

    for (size_t i = 0; i < testData.size(); ++i) {
        string predictedLabel = classifyKNN(trainData, testData[i].features, K);
        string actualLabel = testData[i].label;

        cout << "Test Sample " << i + 1 << ": Actual = " << actualLabel 
             << " | Predicted = " << predictedLabel;

        if (predictedLabel == actualLabel) {
            cout << " [CORRECT]\n";
            correctPredictions++;
        } else {
            cout << " [WRONG]\n";
        }
    }

    double accuracy = (double)correctPredictions / testData.size() * 100.0;
    cout << "--------------------------------------------------------\n";
    cout << fixed << setprecision(2);
    cout << "Model Accuracy: " << accuracy << "%\n";
    cout << "========================================================\n";

    return 0;
}