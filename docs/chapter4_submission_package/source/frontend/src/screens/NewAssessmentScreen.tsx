import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  Eye,
  AlertTriangle,
  CheckCircle2,
  ArrowRight,
  ShieldAlert
} from 'lucide-react';
import { EyeLaterality } from '../types/clinical';
import { analyzeRetinalImageOnCanvas, ClientValidationResult } from '../utils/retinalValidator';

interface NewAssessmentScreenProps {
  onStartValidation: (data: {
    patientId: string;
    laterality: EyeLaterality;
    cameraModel: string;
    isMydriatic: boolean;
    clinicalNotes: string;
    imageDataUrl: string;
    fileSizeBytes: number;
    filename: string;
    clientValidation?: ClientValidationResult;
  }) => void;
  onCancel: () => void;
}

export const NewAssessmentScreen: React.FC<NewAssessmentScreenProps> = ({
  onStartValidation,
  onCancel,
}) => {
  // Patient Context State
  const [patientId, setPatientId] = useState<string>('PT-');
  const [laterality, setLaterality] = useState<EyeLaterality>('OD');
  const [cameraModel, setCameraModel] = useState<string>('Topcon TRC-NW400 Non-Mydriatic');
  const [isMydriatic, setIsMydriatic] = useState<boolean>(false);
  const [clinicalNotes, setClinicalNotes] = useState<string>('');

  // File Upload State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileDimensions, setFileDimensions] = useState<{ width: number; height: number } | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [validationWarning, setValidationWarning] = useState<string | null>(null);
  const [clientValidation, setClientValidation] = useState<ClientValidationResult | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Pre-flight validation logic
  const processImageFile = (file: File) => {
    setFileError(null);
    setValidationWarning(null);
    setClientValidation(null);

    // 1. MIME check
    const validMimes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validMimes.includes(file.type.toLowerCase())) {
      setFileError('Invalid format. DR-CDSS requires high-resolution standard retinal fundus photographs in JPEG or PNG format.');
      return;
    }

    // 2. Size check (<= 15MB)
    const maxBytes = 15 * 1024 * 1024;
    if (file.size > maxBytes) {
      setFileError('File exceeds maximum allowable size (15 MB). Please compress or export at optimal clinical resolution.');
      return;
    }

    setSelectedFile(file);

    // Generate preview & dimension check
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      setPreviewUrl(dataUrl);

      // Check pixel dimensions and run real-time client-side retinal computer vision analysis
      const img = new Image();
      img.onload = () => {
        setFileDimensions({ width: img.width, height: img.height });

        // Run off-screen canvas analysis for circular aperture, chromatic R/B ratio, and blur
        const analysis = analyzeRetinalImageOnCanvas(img, file.size, file.type);
        setClientValidation(analysis);

        if (!analysis.allPassed && analysis.failedGate) {
          if (analysis.failedGate === 2) {
            setValidationWarning(
              analysis.gate2.rejectionReason ||
              'Non-retinal content detected: Image does not exhibit ophthalmic fundus chromatic characteristics (mean R/B ratio < 1.15) or circular aperture. Suspected document, schematic, or non-ocular photography.'
            );
          } else if (analysis.failedGate === 3) {
            setValidationWarning(
              analysis.gate3.rejectionReason ||
              'Image quality warning: low optical sharpness or motion blur detected.'
            );
          } else if (analysis.failedGate === 1) {
            setFileError(analysis.gate1.rejectionReason || 'Invalid file format or size.');
          }
        } else {
          setValidationWarning(null);
        }
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processImageFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!patientId.trim() || patientId === 'PT-') {
      setFileError('Please enter a valid de-identified Study / Patient Identifier.');
      return;
    }

    if (!previewUrl) {
      setFileError('Please upload or select a retinal fundus photograph before proceeding.');
      return;
    }

    onStartValidation({
      patientId: patientId.trim(),
      laterality,
      cameraModel,
      isMydriatic,
      clinicalNotes,
      imageDataUrl: previewUrl,
      fileSizeBytes: selectedFile?.size || 3400000,
      filename: selectedFile?.name || `fundus_${laterality}.jpg`,
      clientValidation: clientValidation || undefined,
    });
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Page Title */}
      <div className="border-b border-slate-200 pb-4 mb-6 flex items-start justify-between">
        <div>
          <h2 className="text-xl font-black text-slate-900 leading-tight">
            New Retinal Fundus Assessment Ingestion
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Capture patient context and upload digital fundus photography for 3-gate validation and preliminary decision support.
          </p>
        </div>
        <span className="inline-flex items-center px-2.5 py-1 rounded bg-teal-50 text-teal-800 text-xs font-mono font-semibold border border-teal-200">
          Screen 3: Ingestion
        </span>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Section 1: Patient Context Form */}
        <section aria-labelledby="patient-context-heading" className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
          <h3 id="patient-context-heading" className="text-xs font-bold uppercase tracking-wider text-slate-600">
            1. Patient & Anatomical Context
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Patient ID */}
            <div>
              <label htmlFor="patient-id" className="block text-xs font-semibold text-slate-700">
                De-Identified Patient / Study ID *
              </label>
              <input
                id="patient-id"
                type="text"
                required
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                placeholder="e.g. PT-94102"
                className="mt-1 block w-full px-3 py-2 text-xs border border-slate-300 rounded-lg shadow-xs focus:ring-2 focus:ring-clinical-primary focus:border-clinical-primary font-mono uppercase"
              />
              <p className="text-[10px] text-slate-400 mt-1">
                Anonymized alphanumeric study identifier ensuring research participant privacy.
              </p>
            </div>

            {/* Eye Laterality Toggle */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Eye Laterality *
              </label>
              <div className="grid grid-cols-2 gap-2" role="radiogroup" aria-label="Eye Laterality Selection">
                <button
                  type="button"
                  onClick={() => setLaterality('OD')}
                  className={`py-2 px-3 text-xs font-bold rounded-lg border flex items-center justify-center gap-2 transition focus-visible:ring-2 focus-visible:ring-clinical-primary ${
                    laterality === 'OD'
                      ? 'bg-teal-50 border-teal-500 text-teal-900 shadow-xs'
                      : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                  role="radio"
                  aria-checked={laterality === 'OD'}
                >
                  <Eye className="w-4 h-4 text-teal-600" />
                  <span>OD (Right Eye)</span>
                </button>

                <button
                  type="button"
                  onClick={() => setLaterality('OS')}
                  className={`py-2 px-3 text-xs font-bold rounded-lg border flex items-center justify-center gap-2 transition focus-visible:ring-2 focus-visible:ring-clinical-primary ${
                    laterality === 'OS'
                      ? 'bg-blue-50 border-blue-500 text-blue-900 shadow-xs'
                      : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                  role="radio"
                  aria-checked={laterality === 'OS'}
                >
                  <Eye className="w-4 h-4 text-blue-600" />
                  <span>OS (Left Eye)</span>
                </button>
              </div>
            </div>

            {/* Camera Model */}
            <div>
              <label htmlFor="camera-model" className="block text-xs font-semibold text-slate-700">
                Fundus Camera Specification
              </label>
              <select
                id="camera-model"
                value={cameraModel}
                onChange={(e) => setCameraModel(e.target.value)}
                className="mt-1 block w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white shadow-xs focus:ring-2 focus:ring-clinical-primary"
              >
                <option value="Topcon TRC-NW400 Non-Mydriatic">Topcon TRC-NW400 (45° Non-Mydriatic)</option>
                <option value="Canon CR-2 AF Digital Retinal Camera">Canon CR-2 AF (45° Non-Mydriatic)</option>
                <option value="Zeiss Visucam 500">Zeiss Visucam 500 (Field 45°/30°)</option>
                <option value="Optos Daytona Ultra-Widefield">Optos Daytona Ultra-Widefield (Standard Crop)</option>
              </select>
            </div>

            {/* Mydriasis checkbox */}
            <div className="flex items-center space-x-2 pt-6">
              <input
                id="mydriatic-toggle"
                type="checkbox"
                checked={isMydriatic}
                onChange={(e) => setIsMydriatic(e.target.checked)}
                className="w-4 h-4 text-clinical-primary rounded border-slate-300 focus:ring-clinical-primary"
              />
              <label htmlFor="mydriatic-toggle" className="text-xs font-medium text-slate-700 cursor-pointer">
                Mydriatic Examination (Pharmacologically dilated pupil)
              </label>
            </div>
          </div>

          {/* Clinical Notes */}
          <div>
            <label htmlFor="clinical-notes" className="block text-xs font-semibold text-slate-700">
              Clinical Context / Examination Indication (Optional)
            </label>
            <textarea
              id="clinical-notes"
              rows={2}
              value={clinicalNotes}
              onChange={(e) => setClinicalNotes(e.target.value)}
              placeholder="e.g. Type 2 DM 10 yrs, HbA1c 8.2%, complaint of floaters in right temporal field..."
              className="mt-1 block w-full px-3 py-2 text-xs border border-slate-300 rounded-lg shadow-xs focus:ring-2 focus:ring-clinical-primary"
            />
          </div>
        </section>

        {/* Section 2: Drag and Drop Retinal Fundus Uploader */}
        <section aria-labelledby="upload-heading" className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 id="upload-heading" className="text-xs font-bold uppercase tracking-wider text-slate-600">
              2. Digital Fundus Photograph Upload
            </h3>
            <span className="text-[11px] text-slate-400 font-mono">
              Accepted: JPEG, PNG • Max: 15MB • Min: 512x512
            </span>
          </div>

          {/* Ingestion Drop Zone */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors duration-150 ${
              isDragging
                ? 'border-clinical-primary bg-clinical-primary-light/40'
                : previewUrl
                ? 'border-teal-400 bg-teal-50/20'
                : 'border-slate-300 hover:border-clinical-primary bg-slate-50/60'
            }`}
            tabIndex={0}
            role="button"
            aria-label="Click or drop retinal photograph file here"
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInputRef.current?.click();
              }
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  processImageFile(e.target.files[0]);
                }
              }}
            />

            {previewUrl ? (
              <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
                <div className="relative w-36 h-36 rounded-full overflow-hidden border-2 border-teal-500 shadow-md bg-black flex-shrink-0">
                  <img
                    src={previewUrl}
                    alt="Fundus preview"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 ring-1 ring-inset ring-black/20 rounded-full" />
                </div>

                <div className="text-left space-y-1 text-xs">
                  {clientValidation && !clientValidation.allPassed ? (
                    <div className="font-bold text-rose-800 flex items-center gap-1.5">
                      <ShieldAlert className="w-4 h-4 text-rose-600" />
                      <span>
                        {clientValidation.failedGate === 2
                          ? 'Non-Retinal Content Detected (Gate 2 Failure)'
                          : clientValidation.failedGate === 3
                          ? 'Quality Issue Detected (Gate 3)'
                          : 'Format / File Issue (Gate 1)'}
                      </span>
                    </div>
                  ) : (
                    <div className="font-bold text-slate-900 flex items-center gap-1.5">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                      <span>File Selected & Pre-flight Passed</span>
                    </div>
                  )}
                  <p className="text-slate-600 font-mono text-[11px]">
                    {selectedFile?.name || 'synthetic_retinal_fundus.jpg'}
                  </p>
                  <p className="text-slate-500 text-[11px]">
                    Size: {selectedFile ? (selectedFile.size / 1024 / 1024).toFixed(2) : '3.4'} MB
                  </p>
                  {fileDimensions && (
                    <p className="text-teal-700 font-mono text-[11px] font-semibold">
                      Dimensions: {fileDimensions.width} × {fileDimensions.height} px
                    </p>
                  )}
                  <p className="text-[11px] text-clinical-primary hover:underline font-semibold pt-1">
                    Click to replace image
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-3 py-4">
                <div className="mx-auto w-12 h-12 rounded-full bg-teal-50 text-clinical-primary flex items-center justify-center">
                  <UploadCloud className="w-6 h-6" />
                </div>
                <div className="text-xs text-slate-600">
                  <span className="font-bold text-clinical-primary">Click to browse</span> or drag and drop retinal fundus photography here
                </div>
                <p className="text-[11px] text-slate-400">
                  Standard posterior pole 45° macular or disc-centered field recommended
                </p>
              </div>
            )}
          </div>

          {/* Validation Pre-Screening Warning Alert */}
          {validationWarning && (
            <div className="p-3.5 bg-rose-50 border-2 border-rose-300 text-rose-900 text-xs rounded-xl flex items-start gap-2.5">
              <ShieldAlert className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
              <div className="space-y-1">
                <span className="font-bold block text-sm">Gate 2 Safeguard Alert: Non-Retinal Modality Detected</span>
                <p className="text-[11px] leading-relaxed font-medium">{validationWarning}</p>
                <p className="text-[10px] text-rose-700 font-mono">
                  Proceeding will trigger the fail-closed validation rejection. Downstream model evaluation and Grad-CAM will be strictly prohibited.
                </p>
              </div>
            </div>
          )}

          {/* File Error Alert */}
          {fileError && (
            <div className="p-3 bg-rose-50 border border-rose-200 text-rose-800 text-xs rounded-lg flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
              <span>{fileError}</span>
            </div>
          )}
        </section>

        {/* Action Controls */}
        <div className="flex items-center justify-between pt-2">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-lg border border-slate-300 transition"
          >
            Cancel & Return to Worklist
          </button>

          <button
            type="submit"
            disabled={!previewUrl}
            className="px-6 py-2.5 text-xs font-bold text-white bg-clinical-primary hover:bg-clinical-primary-hover rounded-lg shadow-sm transition disabled:opacity-50 flex items-center gap-2 focus-visible:ring-2 focus-visible:ring-clinical-primary"
          >
            <span>Proceed to 3-Gate Validation</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </form>
    </div>
  );
};

