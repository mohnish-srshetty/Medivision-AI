import React from 'react';
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Particles } from "./components/magicui/particles";
import { MapPin, Phone, Mail } from 'lucide-react';

const Contact = () => {
  return (
    <div className="relative flex w-full items-center justify-center min-h-screen bg-background px-4 py-12">
      <div className="relative z-10 w-full max-w-4xl">
        <h2 className="text-4xl font-bold mb-12 text-foreground text-center">Contact Us</h2>
        
        <div className="grid md:grid-cols-2 gap-8">
          {/* Contact Information */}
          <Card className="shadow-2xl border border-border">
            <CardContent className="p-8">
              <h3 className="text-2xl font-semibold mb-6">Get in Touch</h3>
              
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <MapPin className="h-6 w-6 text-primary mt-1" />
                  <div>
                    <h4 className="font-semibold mb-1">Address</h4>
                    <p className="text-muted-foreground">
                      MediVision AI<br />
                      Ramaiah University of Applied Sciences<br />
                      Peenya, Bangalore<br />
                      Karnataka 560058
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <Phone className="h-6 w-6 text-primary mt-1" />
                  <div>
                    <h4 className="font-semibold mb-1">Phone</h4>
                    <p className="text-muted-foreground">+91 (080) 1234-5678</p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <Mail className="h-6 w-6 text-primary mt-1" />
                  <div>
                    <h4 className="font-semibold mb-1">Email</h4>
                    <p className="text-muted-foreground">contact@medivision.ai</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Contact Form */}
          <Card className="shadow-2xl border border-border">
            <CardContent className="p-8">
              <h3 className="text-2xl font-semibold mb-6">Send us a Message</h3>
              <form className="space-y-6">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium mb-1 text-foreground">
                    Name
                  </label>
                  <Input id="name" type="text" placeholder="Your Name" />
                </div>
                <div>
                  <label htmlFor="email" className="block text-sm font-medium mb-1 text-foreground">
                    Email
                  </label>
                  <Input id="email" type="email" placeholder="you@example.com" />
                </div>
                <div>
                  <label htmlFor="message" className="block text-sm font-medium mb-1 text-foreground">
                    Message
                  </label>
                  <Textarea id="message" placeholder="Your message..." rows={6} />
                </div>
                <Button type="submit" className="w-full">
                  Send Message
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>

      <Particles
        className="absolute inset-0 z-0"
        quantity={100}
        ease={80}
        refresh
      />
    </div>
  );
};

export default Contact;
