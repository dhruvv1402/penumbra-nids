import type { Metadata } from "next";
import { Anton, Archivo } from "next/font/google";

import Landing from "@/components/landing/Landing";

import "./landing.css";

// Self-hosted by next/font at build time: the browser never asks Google for anything.
const display = Anton({ weight: "400", subsets: ["latin"], variable: "--font-display", display: "swap" });
const body = Archivo({ subsets: ["latin"], variable: "--font-body", display: "swap" });

export const metadata: Metadata = {
  title: "Penumbra — see the unseen",
  description:
    "An ML network detector with two heads: one names the attacks it knows, one flags what it has never " +
    "seen. It alerts a SOC and never blocks traffic.",
};

export default function Home() {
  return (
    <div className={`${display.variable} ${body.variable}`}>
      <Landing />
    </div>
  );
}
