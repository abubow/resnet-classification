import UploadImage from '../components/UploadImage';
import { Separator } from "@/components/ui/separator"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 bg-gray-100">
      <div className="w-full max-w-md p-6 bg-white rounded-lg shadow-md text-slate-900 flex flex-col">
        <h1 className="text-2xl font-bold text-center mb-4">Upload Image for Prediction</h1>
        <Separator className='mb-4' decorative/>
        <UploadImage />
      </div>
    </main>
  );
}
