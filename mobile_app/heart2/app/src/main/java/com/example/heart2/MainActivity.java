package com.example.heart2;

import android.os.Bundle;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.RadioGroup;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.Toast;

import com.google.gson.Gson;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {

    // If running Flask locally on your machine and testing in emulator, use 10.0.2.2
    // Example: "http://10.0.2.2:5000/predict"
    private static final String SERVER_URL = "http://10.0.2.2:5000/predict";

    EditText input_age, input_trestbps, input_chol, input_thalach, input_oldpeak;
    RadioGroup radio_sex;
    Spinner spinner_cp, spinner_fbs, spinner_restecg, spinner_exang, spinner_slope, spinner_ca, spinner_thal;
    Button btn_predict;
    TextView text_result, text_prob, text_error;

    OkHttpClient client;
    Gson gson;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        input_age = findViewById(R.id.input_age);
        input_trestbps = findViewById(R.id.input_trestbps);
        input_chol = findViewById(R.id.input_chol);
        input_thalach = findViewById(R.id.input_thalach);
        input_oldpeak = findViewById(R.id.input_oldpeak);

        radio_sex = findViewById(R.id.radio_sex);

        spinner_cp = findViewById(R.id.spinner_cp);
        spinner_fbs = findViewById(R.id.spinner_fbs);
        spinner_restecg = findViewById(R.id.spinner_restecg);
        spinner_exang = findViewById(R.id.spinner_exang);
        spinner_slope = findViewById(R.id.spinner_slope);
        spinner_ca = findViewById(R.id.spinner_ca);
        spinner_thal = findViewById(R.id.spinner_thal);

        btn_predict = findViewById(R.id.btn_predict);
        text_result = findViewById(R.id.text_result);
        text_prob = findViewById(R.id.text_prob);
        text_error = findViewById(R.id.text_error);

        client = new OkHttpClient();
        gson = new Gson();

        setupSpinners();

        btn_predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                text_error.setText("");
                try {
                    Map<String, Object> data = collectInputs();
                    if (data == null) return; // validation failed; message shown
                    sendPredictionRequest(data);
                } catch (Exception e) {
                    e.printStackTrace();
                    text_error.setText("Error: " + e.getMessage());
                }
            }
        });
    }

    private void setupSpinners() {
        // cp: 0-3
        String[] cpLabels = new String[] {"0: Typical angina", "1: Atypical angina", "2: Non-anginal pain", "3: Asymptomatic"};
        spinner_cp.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, cpLabels));

        // fbs: 0-1
        String[] fbsLabels = new String[] {"0: False", "1: True"};
        spinner_fbs.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, fbsLabels));

        // restecg: 0-2
        String[] restecgLabels = new String[] {"0: Normal", "1: ST-T abnormality", "2: LV hypertrophy"};
        spinner_restecg.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, restecgLabels));

        // exang: 0-1
        String[] exangLabels = new String[] {"0: No", "1: Yes"};
        spinner_exang.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, exangLabels));

        // slope: 0-2
        String[] slopeLabels = new String[] {"0: Upsloping", "1: Flat", "2: Downsloping"};
        spinner_slope.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, slopeLabels));

        // ca: 0-4
        String[] caLabels = new String[] {"0", "1", "2", "3", "4"};
        spinner_ca.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, caLabels));

        // thal: 0-3
        String[] thalLabels = new String[] {"0: Normal", "1: Fixed defect", "2: Reversible defect", "3: Other"};
        spinner_thal.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, thalLabels));
    }

    private Map<String, Object> collectInputs() {
        // basic validation & parse
        String sAge = input_age.getText().toString().trim();
        String sTrest = input_trestbps.getText().toString().trim();
        String sChol = input_chol.getText().toString().trim();
        String sThalach = input_thalach.getText().toString().trim();
        String sOld = input_oldpeak.getText().toString().trim();

        if (sAge.isEmpty() || sTrest.isEmpty() || sChol.isEmpty() || sThalach.isEmpty() || sOld.isEmpty()) {
            text_error.setText("Please fill all numeric fields.");
            return null;
        }

        int age = Integer.parseInt(sAge);
        int trestbps = Integer.parseInt(sTrest);
        int chol = Integer.parseInt(sChol);
        int thalach = Integer.parseInt(sThalach);
        double oldpeak = Double.parseDouble(sOld);

        // Validate numeric ranges (same as your streamlit)
        if (age < 29 || age > 77) { text_error.setText("Age must be 29-77."); return null; }
        if (trestbps < 94 || trestbps > 200) { text_error.setText("trestbps must be 94-200."); return null; }
        if (chol < 126 || chol > 564) { text_error.setText("chol must be 126-564."); return null; }
        if (thalach < 71 || thalach > 202) { text_error.setText("thalach must be 71-202."); return null; }
        if (oldpeak < 0.0 || oldpeak > 6.2) { text_error.setText("oldpeak must be 0.0-6.2."); return null; }

        // sex
        int selectedSexId = radio_sex.getCheckedRadioButtonId();
        RadioButton rb = findViewById(selectedSexId);
        int sex = 0;
        if (rb != null) {
            String sexText = rb.getText().toString().toLowerCase();
            sex = sexText.contains("male") ? 1 : 0;
        }

        // categorical spinners - extract leading integer before colon (or entire string)
        int cp = spinner_cp.getSelectedItemPosition(); // 0..3
        int fbs = spinner_fbs.getSelectedItemPosition(); // 0..1
        int restecg = spinner_restecg.getSelectedItemPosition(); // 0..2
        int exang = spinner_exang.getSelectedItemPosition(); // 0..1
        int slope = spinner_slope.getSelectedItemPosition(); // 0..2
        int ca = spinner_ca.getSelectedItemPosition(); // 0..4
        int thal = spinner_thal.getSelectedItemPosition(); // 0..3

        Map<String, Object> data = new HashMap<>();
        data.put("age", age);
        data.put("trestbps", trestbps);
        data.put("chol", chol);
        data.put("thalach", thalach);
        data.put("oldpeak", oldpeak);
        data.put("sex", sex);
        data.put("cp", cp);
        data.put("fbs", fbs);
        data.put("restecg", restecg);
        data.put("exang", exang);
        data.put("slope", slope);
        data.put("ca", ca);
        data.put("thal", thal);

        return data;
    }

    private void sendPredictionRequest(Map<String, Object> data) {
        String json = gson.toJson(data);
        MediaType JSON = MediaType.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.create(json, JSON);

        Request request = new Request.Builder()
                .url(SERVER_URL)
                .post(body)
                .build();

        // Async call
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                // runs on background thread
                runOnUiThread(() -> text_error.setText("Server error: " + e.getMessage()));
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                final String resp = response.body().string();
                runOnUiThread(() -> {
                    if (!response.isSuccessful()) {
                        text_error.setText("Server returned error: " + response.code() + " - " + resp);
                        return;
                    }
                    try {
                        // expected JSON: {"prediction": 0, "probability": 0.12}
                        Map result = gson.fromJson(resp, Map.class);
                        Double pred = ((Number) result.get("prediction")).doubleValue();
                        Double prob = ((Number) result.get("probability")).doubleValue();
                        if (pred.intValue() == 1) {
                            text_result.setText("Model predicts: HEART DISEASE (class=1)");
                            text_prob.setText(String.format("Probability: %.2f", prob));
                        } else {
                            text_result.setText("Model predicts: NO HEART DISEASE (class=0)");
                            text_prob.setText(String.format("Probability of heart disease: %.2f", prob));
                        }
                    } catch (Exception e) {
                        text_error.setText("Parsing error: " + e.getMessage());
                    }
                });
            }
        });
    }
}