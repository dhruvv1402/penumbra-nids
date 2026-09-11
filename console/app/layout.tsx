import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Penumbra — SOC Console",
  description:
    "ML-native network detection and response. Surfaces alerts for analyst triage; never blocks traffic.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
