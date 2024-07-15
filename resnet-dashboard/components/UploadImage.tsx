"use client";
import React, { useState } from "react";
import axios from "axios";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import Image from "next/image";

const UploadImage = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<{
    prediction: string;
    probability: number;
  } | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedFile) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await axios.post(
        "http://localhost:8000/api/predict/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setPrediction(response.data);
    } catch (error) {
      console.error("Error uploading the file", error);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit} className="grid gap-4">
        <Input type="file" onChange={handleFileChange} />
        {selectedFile && (
          <Image
            src={URL.createObjectURL(selectedFile)}
            alt="Selected image"
            width={300}
            height={300}
          />
        )}
        <Button
          type="submit"
          variant={selectedFile ? "default" : "outline"}
          disabled={!selectedFile}
        >
          Upload
        </Button>
      </form>
      {prediction && (
        <div className="mt-4 p-4 border rounded">
          <h2 className="text-xl font-bold">Prediction</h2>
          <p>Class: {prediction.prediction}</p>
          <p>Probability: {prediction.probability}</p>
        </div>
      )}
    </div>
  );
};

export default UploadImage;
